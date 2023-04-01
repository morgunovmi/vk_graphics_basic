#include "shadowmap_render.h"

#include "../../render/render_gui.h"

void SimpleShadowmapRender::SetupGUIElements()
{
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  {
//    ImGui::ShowDemoWindow();
    ImGui::Begin("Simple render settings");

    ImGui::ColorEdit3("Meshes base color", m_uniforms.baseColor.M, ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_NoInputs);
    ImGui::SliderFloat3("Light source position", m_uniforms.lightPos.M, -10.f, 10.f);
    ImGui::SliderFloat("Min terrain height", &m_noiseConsts.minHeight, -10.f, m_noiseConsts.maxHeight);
    ImGui::SliderFloat("Max terrain height", &m_noiseConsts.maxHeight, m_noiseConsts.minHeight, 10.f);
    ImGui::SliderFloat3("Terrain rotation", m_terrainRotation.M, -90.f, 90.f);
    ImGui::SliderFloat3("Fog box size", m_boxSize.M, 1.f, 30.f);
    ImGui::SliderFloat3("Fog box offset", m_boxOffset.M, -10.f, 10.f);
    ImGui::SliderFloat("Exctinction coef", &m_extinctionCoef, 0.0f, 10.f);
    ImGui::SliderFloat3("Noise scale", m_noiseScale.M, 0.0f, 10.f);
    ImGui::SliderFloat3("Noise offset", m_noiseOffset.M, -20.f, 20.f);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::NewLine();

    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),"Press 'B' to recompile and reload shaders");
    ImGui::End();
  }

  // Rendering
  ImGui::Render();
}
