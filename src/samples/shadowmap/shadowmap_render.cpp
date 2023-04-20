#include "shadowmap_render.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <iostream>

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/RenderTargetStates.hpp>
#include <vulkan/vulkan_core.h>
#include <random>
#include <numeric>


/// RESOURCE ALLOCATION

void SimpleShadowmapRender::AllocateResources()
{
  gbuffer.albedo = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "gbuffer_albedo",
    .format = vk::Format::eR8G8B8A8Srgb,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
  });

  gbuffer.normals = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "gbuffer_normals",
    .format = vk::Format::eR16G16B16A16Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
  });

  gbuffer.positions = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "gbuffer_positions",
    .format = vk::Format::eR16G16B16A16Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
  });

  mainViewDepth = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD16Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled
  });

  shadowMap = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{2048, 2048, 1},
    .name = "shadow_map",
    .format = vk::Format::eD16Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});
  constants = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(UniformParams),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "constants"
  });

  kernelBuffer = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(float4) * ssaoKernelSize,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "ssao_samples"
  });

  noiseBuffer = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = 4 * 4 * 4 * 4, // 4x4 rgba32f image
    .bufferUsage = vk::BufferUsageFlagBits::eTransferSrc,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "noiseStagingBuffer"
  });
  noiseTexture = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{4, 4, 1},
    .name = "noise_texture",
    .format = vk::Format::eR32G32B32A32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled
  });
  noiseTextureSampler = etna::Sampler(etna::Sampler::CreateInfo{.addressMode = vk::SamplerAddressMode::eRepeat, .name = "noise_sampler"});

  ssaoRawImage = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "ssao_raw",
    .format = vk::Format::eR16Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
  });

  ssaoBlurredImage = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "ssao_blurred",
    .format = vk::Format::eR16Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
  });

  m_uboMappedMem = constants.map();
}

void SimpleShadowmapRender::generateSsaoKernel()
{
  std::random_device rd{};
  std::default_random_engine gen{rd()};
  std::uniform_real_distribution<float> randomFloats{0.0f, 1.0f};

  ssaoKernel.clear();
  for (size_t i = 0; i < ssaoKernelSize; ++i)
  {
    LiteMath::float4 sample{
      randomFloats(gen) * 2.0f - 1.0f,
      randomFloats(gen) * 2.0f - 1.0f,
      randomFloats(gen),
      0.0f
    };
    sample = LiteMath::normalize(sample);
    sample *= randomFloats(gen);

    float scale = i / static_cast<float>(ssaoKernelSize); 
    scale = LiteMath::lerp(0.1f, 1.0f, scale * scale);
    sample *= scale;
    ssaoKernel.push_back(sample);
  }
  auto *mapped_mem = kernelBuffer.map();
  memcpy(mapped_mem, ssaoKernel.data(), sizeof(float4) * ssaoKernel.size());
  kernelBuffer.unmap();
}

void SimpleShadowmapRender::LoadScene(const char* path, bool transpose_inst_matrices)
{
  m_pScnMgr->LoadSceneXML(path, transpose_inst_matrices);

  // TODO: Make a separate stage
  loadShaders();
  PreparePipelines();

  auto loadedCam = m_pScnMgr->GetCamera(0);
  m_cam.fov = loadedCam.fov;
  m_cam.pos = float3(loadedCam.pos);
  m_cam.up  = float3(loadedCam.up);
  m_cam.lookAt = float3(loadedCam.lookAt);
  m_cam.tdist  = loadedCam.farPlane;

  std::random_device rd{};
  std::default_random_engine gen{rd()};
  std::uniform_real_distribution<float> colorDistr{0.0f, 1.0f};
  objColors.clear();
  for (size_t i = 0; i < m_pScnMgr->InstancesNum(); ++i)
  {
    objColors.emplace_back(LiteMath::float4{colorDistr(gen), colorDistr(gen), colorDistr(gen), 1.0f});
  }

  std::uniform_real_distribution<float> randomFloats{0.0f, 1.0f};
  generateSsaoKernel();

  ssaoNoise.clear();
  for (size_t i = 0; i < 16; ++i)
  {
    ssaoNoise.emplace_back(
      randomFloats(gen) * 2.0f - 1.0f,
      randomFloats(gen) * 2.0f - 1.0f,
      0.0f,
      0.0f
    );
  }
  m_noiseBufferMappedMep = noiseBuffer.map();
  memcpy(m_noiseBufferMappedMep, ssaoNoise.data(), sizeof(float4) * ssaoNoise.size());
  noiseBuffer.unmap();

  copyNoise();

  m_coeffs.resize(m_kernelSize);
  const float kernelRadius = (m_kernelSize - 1) / 2.f;
  const auto sigma = kernelRadius / 3.f;
  const auto sigma2 = 2 * sigma * sigma;
  for (size_t i = 0; i < m_kernelSize; ++i)
  {
    const auto delta = static_cast<float>(i) - kernelRadius;
    m_coeffs[i] = std::exp(-delta * delta / sigma2);
  }
  const auto sum = std::accumulate(m_coeffs.cbegin(), m_coeffs.cend(), 0.f);
  for (size_t i = 0; i < m_kernelSize; ++i)
  {
    m_coeffs[i] /= sum;
  }
}

void SimpleShadowmapRender::copyNoise()
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(m_cmdBufferAux, &beginInfo);

    // commands
    etna::set_state(m_cmdBufferAux, noiseTexture.get(), vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eTransferDstOptimal,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(m_cmdBufferAux);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {4, 4, 1};

    vkCmdCopyBufferToImage(
        m_cmdBufferAux,
        noiseBuffer.get(),
        noiseTexture.get(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    etna::set_state(m_cmdBufferAux, noiseTexture.get(), vk::PipelineStageFlagBits2::eFragmentShader,
      vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(m_cmdBufferAux);

    vkEndCommandBuffer(m_cmdBufferAux);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_cmdBufferAux;

    vkQueueSubmit(m_context->getQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_context->getQueue());
}

void SimpleShadowmapRender::DeallocateResources()
{
  gbuffer.albedo.reset();
  gbuffer.normals.reset();
  mainViewDepth.reset(); // TODO: Make an etna method to reset all the resources
  shadowMap.reset();
  m_swapchain.Cleanup();
  vkDestroySurfaceKHR(GetVkInstance(), m_surface, nullptr);  

  constants = etna::Buffer();
}





/// PIPELINES CREATION

void SimpleShadowmapRender::PreparePipelines()
{
  // create full screen quad for debug purposes
  // 
  m_pFSQuad = std::make_shared<vk_utils::QuadRenderer>(0,0, 512, 512);
  m_pFSQuad->Create(m_context->getDevice(),
    VK_GRAPHICS_BASIC_ROOT "/resources/shaders/quad3_vert.vert.spv",
    VK_GRAPHICS_BASIC_ROOT "/resources/shaders/quad.frag.spv",
    vk_utils::RenderTargetInfo2D{
      .size          = VkExtent2D{ m_width, m_height },// this is debug full screen quad
      .format        = m_swapchain.GetFormat(),
      .loadOp        = VK_ATTACHMENT_LOAD_OP_LOAD,// seems we need LOAD_OP_LOAD if we want to draw quad to part of screen
      .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .finalLayout   = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL 
    }
  );
  SetupSimplePipeline();
}

void SimpleShadowmapRender::loadShaders()
{
  etna::create_program("simple_shadow", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
  etna::create_program("simple_geometry",
    {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple_geometry.frag.spv", VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
  etna::create_program("simple_deferred", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple_quad.vert.spv",
                                          VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple_deferred.frag.spv"});
  etna::create_program("simple_ssao", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple_quad.vert.spv",
                                      VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple_ssao.frag.spv"});
  etna::create_program("gaussian_compute_vertical", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/gaussian_vertical.comp.spv"});
  etna::create_program("gaussian_compute_horizontal", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/gaussian_horizontal.comp.spv"});
}

void SimpleShadowmapRender::SetupSimplePipeline()
{
  std::vector<std::pair<VkDescriptorType, uint32_t> > dtypes = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,     2}
  };

  m_pBindings = std::make_shared<vk_utils::DescriptorMaker>(m_context->getDevice(), dtypes, 2);
  
  m_pBindings->BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT);
  m_pBindings->BindImage(0, ssaoRawImage.getView({}), defaultSampler.get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  m_pBindings->BindEnd(&m_quadDS, &m_quadDSLayout);

  etna::VertexShaderInputDescription sceneVertexInputDesc
    {
      .bindings = {etna::VertexShaderInputDescription::Binding
        {
          .byteStreamDescription = m_pScnMgr->GetVertexStreamDescription()
        }}
    };

  auto& pipelineManager = etna::get_context().getPipelineManager();
  m_shadowPipeline = pipelineManager.createGraphicsPipeline("simple_shadow",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .fragmentShaderOutput =
        {
          .depthAttachmentFormat = vk::Format::eD16Unorm
        }
    });

  std::vector<vk::PipelineColorBlendAttachmentState> colorAttachmentStates;
  auto blendState = vk::PipelineColorBlendAttachmentState{
    .blendEnable    = false,
    .colorWriteMask = vk::ColorComponentFlagBits::eR
                      | vk::ColorComponentFlagBits::eG
                      | vk::ColorComponentFlagBits::eB
                      | vk::ColorComponentFlagBits::eA
  };
  for (size_t i = 0; i < 3; i++) {
    colorAttachmentStates.push_back(blendState);
  }

  m_geometryPipeline = pipelineManager.createGraphicsPipeline("simple_geometry",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .blendingConfig = colorAttachmentStates,
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {vk::Format::eR8G8B8A8Srgb, vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16B16A16Sfloat},
          .depthAttachmentFormat = vk::Format::eD16Unorm
        }
    });
  m_shadingPipeline = pipelineManager.createGraphicsPipeline("simple_deferred",
    {
      .depthConfig = 
        {
          .depthTestEnable = false,
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {static_cast<vk::Format>(m_swapchain.GetFormat())},
        }
    });
  m_ssaoPipeline = pipelineManager.createGraphicsPipeline("simple_ssao",
    {
      .depthConfig = 
        {
          .depthTestEnable = false,
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {vk::Format::eR16Sfloat},
        }
    });
  m_computePipelineVertical = pipelineManager.createComputePipeline("gaussian_compute_vertical", {});
  m_computePipelineHorizontal = pipelineManager.createComputePipeline("gaussian_compute_horizontal", {});
}

void SimpleShadowmapRender::DestroyPipelines()
{
  m_pFSQuad     = nullptr; // smartptr delete it's resources
}



/// COMMAND BUFFER FILLING

void SimpleShadowmapRender::DrawSceneCmd(VkCommandBuffer a_cmdBuff, const float4x4& a_wvp)
{
  VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

  VkDeviceSize zero_offset = 0u;
  VkBuffer vertexBuf = m_pScnMgr->GetVertexBuffer();
  VkBuffer indexBuf  = m_pScnMgr->GetIndexBuffer();
  
  vkCmdBindVertexBuffers(a_cmdBuff, 0, 1, &vertexBuf, &zero_offset);
  vkCmdBindIndexBuffer(a_cmdBuff, indexBuf, 0, VK_INDEX_TYPE_UINT32);

  pushConst2M.projView = a_wvp;
  for (uint32_t i = 0; i < m_pScnMgr->InstancesNum(); ++i)
  {
    auto inst         = m_pScnMgr->GetInstanceInfo(i);
    auto model = m_pScnMgr->GetInstanceMatrix(i);

    pushConst2M.modelRow1 = model.get_row(0);
    pushConst2M.modelRow2 = model.get_row(1);
    pushConst2M.modelRow3 = model.get_row(2);
    pushConst2M.objColor = objColors[i];
    vkCmdPushConstants(a_cmdBuff, m_geometryPipeline.getVkPipelineLayout(),
      stageFlags, 0, sizeof(pushConst2M), &pushConst2M);

    auto mesh_info = m_pScnMgr->GetMeshInfo(inst.mesh_id);
    vkCmdDrawIndexed(a_cmdBuff, mesh_info.m_indNum, 1, mesh_info.m_indexOffset, mesh_info.m_vertexOffset, 0);
  }
}

void SimpleShadowmapRender::BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, VkImage a_targetImage, VkImageView a_targetImageView)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  //// draw scene to shadowmap
  //
  {
    etna::RenderTargetState renderTargets(a_cmdBuff, {2048, 2048}, {}, shadowMap);

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline.getVkPipeline());
    DrawSceneCmd(a_cmdBuff, m_lightMatrix);
  }

  //// draw positions and normals to gbuffer
  //
  {
    auto geometryInfo  = etna::get_shader_program("simple_geometry");

    auto set = etna::create_descriptor_set(geometryInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding {0, constants.genBinding()},
    });

    VkDescriptorSet vkSet = set.getVkSet();
    etna::RenderTargetState renderTargets(a_cmdBuff, {m_width, m_height},
             {{gbuffer.albedo, gbuffer.normals, gbuffer.positions}}, mainViewDepth);

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_geometryPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_geometryPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    DrawSceneCmd(a_cmdBuff, m_worldViewProj);
  }

  if (ssaoEnabled)
  {
    etna::set_state(a_cmdBuff, noiseTexture.get(), vk::PipelineStageFlagBits2::eFragmentShader,
      vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);
    etna::set_state(a_cmdBuff, gbuffer.normals.get(), vk::PipelineStageFlagBits2::eFragmentShader,
      vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);
    etna::set_state(a_cmdBuff, gbuffer.positions.get(), vk::PipelineStageFlagBits2::eFragmentShader,
      vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(a_cmdBuff);

    //// calculate raw ssao
    //
    {
      auto simpleSsaoInfo  = etna::get_shader_program("simple_ssao");

      auto set = etna::create_descriptor_set(simpleSsaoInfo.getDescriptorLayoutId(0), a_cmdBuff,
      {
        etna::Binding {0, kernelBuffer.genBinding()},
        etna::Binding {1, constants.genBinding()},
        etna::Binding {2, noiseTexture.genBinding(noiseTextureSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding {3, gbuffer.normals.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding {4, gbuffer.positions.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)}
      });

      VkDescriptorSet vkSet = set.getVkSet();

      etna::RenderTargetState renderTargets(a_cmdBuff, {m_width, m_height}, {ssaoRawImage}, {});

      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ssaoPipeline.getVkPipeline());
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_ssaoPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

      vkCmdPushConstants(a_cmdBuff, m_ssaoPipeline.getVkPipelineLayout(),
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstDeferred), &pushConstDeferred);

      vkCmdDraw(a_cmdBuff, 3, 1, 0, 0);
    }

    etna::set_state(a_cmdBuff, ssaoRawImage.get(), vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);
    etna::set_state(a_cmdBuff, ssaoBlurredImage.get(), vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    etna::flush_barriers(a_cmdBuff);

      // Apply horizontal gaussian filter in compute shader
    {
      auto gaussianComputeInfo = etna::get_shader_program("gaussian_compute_horizontal");

      auto set = etna::create_descriptor_set(gaussianComputeInfo.getDescriptorLayoutId(0), a_cmdBuff,
      {
        etna::Binding {0, ssaoRawImage.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral)},
        etna::Binding {1, ssaoBlurredImage.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral)},
      });

      VkDescriptorSet vkSet = set.getVkSet();

      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineHorizontal.getVkPipeline());
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_computePipelineHorizontal.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

      vkCmdPushConstants(a_cmdBuff, m_computePipelineHorizontal.getVkPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                        0, m_coeffs.size() * sizeof(float), m_coeffs.data());

      vkCmdDispatch(a_cmdBuff, 1024 / 32 + 1, 1024, 1);
    }

    etna::set_state(a_cmdBuff, ssaoRawImage.get(), vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);
    etna::set_state(a_cmdBuff, ssaoBlurredImage.get(), vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eGeneral,
      vk::ImageAspectFlagBits::eColor);

    etna::flush_barriers(a_cmdBuff);

      // Apply vertical gaussian filter in compute shader
    {
      auto gaussianComputeInfo = etna::get_shader_program("gaussian_compute_vertical");

      auto set = etna::create_descriptor_set(gaussianComputeInfo.getDescriptorLayoutId(0), a_cmdBuff,
      {
        etna::Binding {0, ssaoBlurredImage.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral)},
        etna::Binding {1, ssaoRawImage.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral)},
      });

      VkDescriptorSet vkSet = set.getVkSet();

      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineVertical.getVkPipeline());
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_computePipelineVertical.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

      vkCmdPushConstants(a_cmdBuff, m_computePipelineVertical.getVkPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                        0, m_coeffs.size() * sizeof(float), m_coeffs.data());

      vkCmdDispatch(a_cmdBuff, 1024, 1024 / 32 + 1, 1);
    }
  }

  etna::set_state(a_cmdBuff, shadowMap.get(), vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eDepth);
  etna::set_state(a_cmdBuff, mainViewDepth.get(), vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eDepth);
  etna::set_state(a_cmdBuff, gbuffer.albedo.get(), vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::set_state(a_cmdBuff, gbuffer.normals.get(), vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::set_state(a_cmdBuff, gbuffer.positions.get(), vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::set_state(a_cmdBuff, ssaoRawImage.get(), vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(a_cmdBuff);

  //// draw final scene to screen
  //
  {
    auto simpleDeferredInfo  = etna::get_shader_program("simple_deferred");

    auto set = etna::create_descriptor_set(simpleDeferredInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding {0, constants.genBinding()},
      etna::Binding {1, shadowMap.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding {2, gbuffer.albedo.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding {3, gbuffer.normals.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding {4, gbuffer.positions.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding {5, ssaoRawImage.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    });

    VkDescriptorSet vkSet = set.getVkSet();

    etna::RenderTargetState renderTargets(a_cmdBuff, {m_width, m_height}, {{a_targetImage, a_targetImageView}}, {});

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadingPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_shadingPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    vkCmdPushConstants(a_cmdBuff, m_shadingPipeline.getVkPipelineLayout(),
      VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstDeferred), &pushConstDeferred);

    vkCmdDraw(a_cmdBuff, 3, 1, 0, 0);
  }

  if(m_input.drawFSQuad)
  {
    float scaleAndOffset[4] = {0.5f, 0.5f, -0.5f, +0.5f};
    m_pFSQuad->SetRenderTarget(a_targetImageView);
    m_pFSQuad->DrawCmd(a_cmdBuff, m_quadDS, scaleAndOffset);
  }

  etna::set_state(a_cmdBuff, a_targetImage, vk::PipelineStageFlagBits2::eBottomOfPipe,
    vk::AccessFlags2(), vk::ImageLayout::ePresentSrcKHR,
    vk::ImageAspectFlagBits::eColor);

  etna::finish_frame(a_cmdBuff);

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}
