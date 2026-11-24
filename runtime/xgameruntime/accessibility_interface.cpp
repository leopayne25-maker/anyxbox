// XAccessibility : mode de contraste élevé réel de Windows.
//
// Méthodes : gdk_api_table.inc. Signature et énumération : Microsoft Learn,
// https://learn.microsoft.com/gaming/gdk/docs/reference/system/xaccessibility/functions/xhighcontrastgetmode
// et .../xaccessibility/enums/xhighcontrastmode (Off = 0, Dark = 1, Light = 2, Other = 3).
// Le mode vient de SystemParametersInfo(SPI_GETHIGHCONTRAST) ; sombre ou clair selon la couleur
// de fond des fenêtres du thème actif. Sous-titres, synthèse vocale et transcription restent
// non écrits : ni leurs réglages ni leurs voix ne sont inventés.
#include <windows.h>

#include <cstdint>

#include "interface_helpers.h"
#include "interfaces.h"
#include "trace.h"

namespace {

enum HighContrastMode : uint32_t { kOff = 0, kDark = 1, kLight = 2 };

// Luminance relative (Rec. 709, 0 à 255) au-dessous de laquelle un fond est sombre.
constexpr double kDarkBackground = 128.0;

bool DarkBackground() {
  const COLORREF window = GetSysColor(COLOR_WINDOW);
  const double luminance = 0.2126 * GetRValue(window) + 0.7152 * GetGValue(window) + 0.0722 * GetBValue(window);
  return luminance < kDarkBackground;
}

HRESULT XHighContrastGetMode(uint32_t* mode) noexcept {
  if (mode == nullptr) return E_INVALIDARG;
  HIGHCONTRASTW contrast{};
  contrast.cbSize = sizeof contrast;
  if (!SystemParametersInfoW(SPI_GETHIGHCONTRAST, sizeof contrast, &contrast, 0)) {
    return HRESULT_FROM_WIN32(GetLastError());
  }
  if ((contrast.dwFlags & HCF_HIGHCONTRASTON) == 0) {
    *mode = kOff;
  } else {
    *mode = DarkBackground() ? kDark : kLight;
  }
  ANYXBOXX_TRACE("[runtime] XHighContrastGetMode : %u", *mode);
  return S_OK;
}

// Méthodes 3 à 21 de l'interface {D722B373…} (gdk_api_table.inc).
void* const kAccessibilityMethods[] = {
    ANYXBOXX_IUNKNOWN_METHODS,              // 0-2
    ANYXBOXX_NOT_IMPLEMENTED,  // 3 XClosedCaptionGetProperties
    ANYXBOXX_NOT_IMPLEMENTED,  // 4 XClosedCaptionSetEnabled
    ANYXBOXX_METHOD(XHighContrastGetMode),  // 5
    ANYXBOXX_NOT_IMPLEMENTED,  // 6 XSpeechToTextSetPositionHint
    ANYXBOXX_NOT_IMPLEMENTED,  // 7 XSpeechToTextSendString
    ANYXBOXX_NOT_IMPLEMENTED,  // 8 XSpeechSynthesizerEnumerateInstalledVoices
    ANYXBOXX_NOT_IMPLEMENTED,  // 9 XSpeechSynthesizerCreate
    ANYXBOXX_NOT_IMPLEMENTED,  // 10 XSpeechSynthesizerCloseHandle
    ANYXBOXX_NOT_IMPLEMENTED,  // 11 XSpeechSynthesizerSetDefaultVoice
    ANYXBOXX_NOT_IMPLEMENTED,  // 12 XSpeechSynthesizerSetCustomVoice
    ANYXBOXX_NOT_IMPLEMENTED,  // 13 XSpeechSynthesizerCreateStreamFromText
    ANYXBOXX_NOT_IMPLEMENTED,  // 14 XSpeechSynthesizerCloseStreamHandle
    ANYXBOXX_NOT_IMPLEMENTED,  // 15 XSpeechSynthesizerGetStreamDataSize
    ANYXBOXX_NOT_IMPLEMENTED,  // 16 XSpeechSynthesizerGetStreamData
    ANYXBOXX_NOT_IMPLEMENTED,  // 17 XSpeechToTextBeginHypothesisString
    ANYXBOXX_NOT_IMPLEMENTED,  // 18 XSpeechToTextUpdateHypothesisString
    ANYXBOXX_NOT_IMPLEMENTED,  // 19 XSpeechToTextFinalizeHypothesisString
    ANYXBOXX_NOT_IMPLEMENTED,  // 20 XSpeechToTextCancelHypothesisString
    ANYXBOXX_NOT_IMPLEMENTED,  // 21 XSpeechSynthesizerCreateStreamFromSsml
};

const anyxboxx::StaticInterface kAccessibility{kAccessibilityMethods};

}  // namespace

namespace anyxboxx {

void* AccessibilityInterface() { return const_cast<StaticInterface*>(&kAccessibility); }

}  // namespace anyxboxx
