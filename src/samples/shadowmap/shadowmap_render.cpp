#include "shadowmap_render.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <iostream>
#include <LiteMath.h>

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/RenderTargetStates.hpp>
#include <vulkan/vulkan_core.h>


/// RESOURCE ALLOCATION

void SimpleShadowmapRender::AllocateResources()
{
  mainViewDepth = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment
  });

  shadowMap = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{2048, 2048, 1},
    .name = "shadow_map",
    .format = vk::Format::eD16Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled
  });

  heightMap = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{2048, 2048, 1},
    .name = "shadow_map",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
  });

  defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});
  constants = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(UniformParams),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "constants"
  });
  quadIndexBuffer = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(uint16_t) * 4,
    .bufferUsage = vk::BufferUsageFlagBits::eIndexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
    .name = "quad_index_buffer"
  });

  m_uboMappedMem = constants.map();
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

  auto mapped_mem = quadIndexBuffer.map();
  std::array<uint16_t, 6> indices{0, 3, 1, 2};
  memcpy(mapped_mem, indices.data(), sizeof(uint16_t) * indices.size());
  quadIndexBuffer.unmap();

  m_terrainMatrix = translate4x4(float3{0, -1, -2}) 
                    * rotate4x4X(DEG_TO_RAD * m_terrainRotation.x)
                    * rotate4x4Y(DEG_TO_RAD * m_terrainRotation.y)
                    * rotate4x4Z(DEG_TO_RAD * m_terrainRotation.z)
                    * rotate4x4X(-M_PI / 2);
}

void SimpleShadowmapRender::DeallocateResources()
{
  mainViewDepth.reset(); // TODO: Make an etna method to reset all the resources
  shadowMap.reset();
  heightMap.reset();
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
  etna::create_program("simple_noise", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/noise.frag.spv",
                                        VK_GRAPHICS_BASIC_ROOT"/resources/shaders/noise_quad.vert.spv"});
  etna::create_program("simple_terrain", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/terrain.frag.spv",
                                        VK_GRAPHICS_BASIC_ROOT"/resources/shaders/terrain.vert.spv",
                                        VK_GRAPHICS_BASIC_ROOT"/resources/shaders/terrain.tesc.spv",
                                        VK_GRAPHICS_BASIC_ROOT"/resources/shaders/terrain.tese.spv"});
  etna::create_program("simple_shadow", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/terrain.vert.spv",
                                        VK_GRAPHICS_BASIC_ROOT"/resources/shaders/terrain.tesc.spv",
                                        VK_GRAPHICS_BASIC_ROOT"/resources/shaders/terrain.tese.spv"});
}

void SimpleShadowmapRender::SetupSimplePipeline()
{
  std::vector<std::pair<VkDescriptorType, uint32_t> > dtypes = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             1},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,     2}
  };

  m_pBindings = std::make_shared<vk_utils::DescriptorMaker>(m_context->getDevice(), dtypes, 2);
  
  m_pBindings->BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT);
  m_pBindings->BindImage(0, shadowMap.getView({}), defaultSampler.get(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  m_pBindings->BindEnd(&m_quadDS, &m_quadDSLayout);

  etna::VertexShaderInputDescription sceneVertexInputDesc
    {
      .bindings = {etna::VertexShaderInputDescription::Binding
        {
          .byteStreamDescription = m_pScnMgr->GetVertexStreamDescription()
        }}
    };

  auto& pipelineManager = etna::get_context().getPipelineManager();
  m_noisePipeline = pipelineManager.createGraphicsPipeline("simple_noise",
    {
      .depthConfig = 
        {
          .depthTestEnable = false,
          .depthWriteEnable = false,
        },
      .fragmentShaderOutput = 
        {
          .colorAttachmentFormats = {vk::Format::eR32Sfloat}
        }
    });
  m_terrainPipeline = pipelineManager.createGraphicsPipeline("simple_terrain",
    {
      .inputAssemblyConfig =
        {
          .topology = vk::PrimitiveTopology::ePatchList,
        },
      .rasterizationConfig =
        {
          .polygonMode = vk::PolygonMode::eFill,
          .lineWidth = 1.0f,
        },
      .tesselationConfig =
        {
          .patchControlPoints = 4,
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {static_cast<vk::Format>(m_swapchain.GetFormat())},
          .depthAttachmentFormat = vk::Format::eD32Sfloat
        }
    });
  m_shadowPipeline = pipelineManager.createGraphicsPipeline("simple_shadow",
    {
      .inputAssemblyConfig =
        {
          .topology = vk::PrimitiveTopology::ePatchList,
        },
      .tesselationConfig =
        {
          .patchControlPoints = 4,
        },
      .fragmentShaderOutput =
        {
          .depthAttachmentFormat = vk::Format::eD16Unorm
        }
    });
}

void SimpleShadowmapRender::DestroyPipelines()
{
  m_pFSQuad     = nullptr; // smartptr delete it's resources
}



/// COMMAND BUFFER FILLING

void SimpleShadowmapRender::DrawSceneCmd(VkCommandBuffer a_cmdBuff, const float4x4& a_wvp)
{
  VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);

  vkCmdBindIndexBuffer(a_cmdBuff, quadIndexBuffer.get(), 0, VK_INDEX_TYPE_UINT16);

  pushConst2M.projView = a_wvp;
  pushConst2M.model = m_terrainMatrix;
  vkCmdPushConstants(a_cmdBuff, m_terrainPipeline.getVkPipelineLayout(),
    stageFlags, 0, sizeof(pushConst2M), &pushConst2M);

  vkCmdDrawIndexed(a_cmdBuff, 4, 1, 0, 0, 0);
}

void SimpleShadowmapRender::BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, VkImage a_targetImage, VkImageView a_targetImageView)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  //// draw noise to texture
  //
  {
    etna::RenderTargetState renderTargets(a_cmdBuff, {2048, 2048}, {heightMap}, {});

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_noisePipeline.getVkPipeline());
    vkCmdPushConstants(a_cmdBuff, m_noisePipeline.getVkPipelineLayout(),
      VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(m_noiseConsts), &m_noiseConsts);
    vkCmdDraw(a_cmdBuff, 3, 1, 0, 0);
  }

  etna::set_state(a_cmdBuff, heightMap.get(), vk::PipelineStageFlagBits2::eTessellationEvaluationShader,
    vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);

  //// draw scene to shadowmap
  //
  {
    auto simpleShadowInfo = etna::get_shader_program("simple_shadow");

    auto set = etna::create_descriptor_set(simpleShadowInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding {0, heightMap.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    });

    VkDescriptorSet vkSet = set.getVkSet();

    etna::RenderTargetState renderTargets(a_cmdBuff, {2048, 2048}, {}, shadowMap);

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_shadowPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);
    DrawSceneCmd(a_cmdBuff, m_lightMatrix);
  }

  //// draw final scene to screen
  //
  {
    auto simpleMaterialInfo = etna::get_shader_program("simple_terrain");

    auto set = etna::create_descriptor_set(simpleMaterialInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding {0, heightMap.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding {1, constants.genBinding()},
      etna::Binding {2, shadowMap.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)}
    });

    VkDescriptorSet vkSet = set.getVkSet();

    etna::RenderTargetState renderTargets(a_cmdBuff, {m_width, m_height}, {{a_targetImage, a_targetImageView}}, mainViewDepth);

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_terrainPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_terrainPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    DrawSceneCmd(a_cmdBuff, m_worldViewProj);
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
