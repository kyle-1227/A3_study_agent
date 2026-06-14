---
name: A3 Study Agent
description: 高校个性化学习资源生成工作台
colors:
  primary: "#35593F"
  primary-deep: "#284531"
  primary-soft: "#E7F0E8"
  background: "#F7F6ED"
  surface: "#FFFFFF"
  surface-muted: "#F0EFE4"
  border: "#DDD9C8"
  text: "#243027"
  text-muted: "#667060"
  accent: "#C9824A"
  success: "#3E7A4A"
  warning: "#B7791F"
  danger: "#C55447"
  info: "#3E6F99"
typography:
  title:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 650
    lineHeight: 1.35
  body:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.65
  label:
    fontFamily: "Geist, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.35
  mono:
    fontFamily: "Geist Mono, Consolas, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.45
rounded:
  sm: "6px"
  md: "8px"
  lg: "12px"
  xl: "16px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
    rounded: "{rounded.md}"
    padding: "8px 14px"
  panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.lg}"
    padding: "16px"
  input:
    backgroundColor: "{colors.surface-muted}"
    textColor: "{colors.text}"
    rounded: "{rounded.xl}"
    padding: "12px 16px"
---

# Design System: A3 Study Agent

## 1. Overview

**Creative North Star: "Transparent Study Workbench"**

A3 Study Agent should feel like a focused academic workspace: a quiet chat surface in the center, navigation memory on the left, and an execution cockpit on the right. The interface is allowed to be dense because the product is technical, but density must be organized through clear labels, stable spacing, and consistent status patterns.

The visual system rejects generic AI-chat spectacle. No purple glow, no decorative glass panels, no fake dashboards, and no ornamental motion. The craft is in making complex agent execution feel readable, not in making the screen look louder.

**Key Characteristics:**
- Restrained green primary color for navigation, active state, and primary actions.
- Warm neutral surfaces for long study sessions, with enough contrast for small status text.
- Flat-by-default panels with thin borders and tonal layering instead of heavy shadows.
- Lucide icons only, with consistent stroke and sizing.
- Chinese-first product copy, concise and operational.

## 2. Colors

The palette is a deep academic green system with neutral study-desk surfaces and small semantic accents.

### Primary
- **Academic Green** (`#35593F`): primary actions, active navigation, graph success states, and brand surfaces.
- **Deep Academic Green** (`#284531`): hover states and high-emphasis text on light green surfaces.
- **Soft Academic Green** (`#E7F0E8`): selected rows, assistant avatar backgrounds, and low-emphasis success surfaces.

### Secondary
- **Measured Amber** (`#C9824A`): running states, wait states, and attention that is not an error.

### Neutral
- **Study Canvas** (`#F7F6ED`): application background.
- **Clean Surface** (`#FFFFFF`): message cards, panels, popovers, and document cards.
- **Muted Surface** (`#F0EFE4`): input areas, graph canvas, and low-emphasis empty states.
- **Soft Border** (`#DDD9C8`): all default borders and dividers.
- **Ink Text** (`#243027`): primary text.
- **Muted Ink** (`#667060`): secondary text and helper copy.

### Named Rules
**The One Accent Rule.** Green carries identity and interaction. Amber appears only for running, waiting, or caution states.

## 3. Typography

**Display Font:** Geist, system-ui fallback  
**Body Font:** Geist, system-ui fallback  
**Label/Mono Font:** Geist Mono, Consolas fallback

**Character:** The type system is compact and product-native. It prioritizes scanability over editorial contrast.

### Hierarchy
- **Title** (650, 16px, 1.35): app name, panel headings, card headings.
- **Body** (400, 14px, 1.65): chat content, explanations, region copy.
- **Small Body** (400, 13px, 1.55): sidebar items, resource details, helper copy.
- **Label** (600, 12px, 1.35): section labels, tabs, chips, compact controls.
- **Mono** (400, 12px, 1.45): logs, token counts, timings.

### Named Rules
**The Compact Clarity Rule.** Product labels stay compact, but never so small that status text becomes decorative.

## 4. Elevation

The system is flat by default. Depth is conveyed through borders, tonal surfaces, and small hover changes rather than large shadows. Shadows are reserved for popovers and dropdowns that must sit above scrollable containers.

### Shadow Vocabulary
- **Popover Lift** (`0 12px 30px rgba(36, 48, 39, 0.12)`): menus, region selectors, and floating tool panels only.

### Named Rules
**The Flat Workbench Rule.** If a surface is part of the permanent app shell, use a border and tonal background instead of a drop shadow.

## 5. Components

### Buttons
- **Shape:** medium rounded rectangle for standard buttons (8px), full pill only for circular icon controls and chat send actions.
- **Primary:** Academic Green background with white text.
- **Hover / Focus:** slightly deeper green on hover, visible focus ring using the primary token at 30% opacity.
- **Ghost:** transparent at rest, soft green surface on hover, never decorative.

### Chips
- **Style:** low-contrast tinted backgrounds with readable text.
- **State:** selected chips use Soft Academic Green with Academic Green text.

### Cards / Containers
- **Corner Style:** 12px for panels and cards, 16px for larger chat/resource containers.
- **Background:** Clean Surface on Study Canvas.
- **Shadow Strategy:** no shadow except popovers.
- **Border:** Soft Border, 1px.
- **Internal Padding:** 12px for compact status cards, 16px for panels, 24px for empty states.

### Inputs / Fields
- **Style:** Muted Surface background, Soft Border stroke, 16px radius for chat composer.
- **Focus:** primary ring and border shift.
- **Error / Disabled:** danger text plus semantic icon, not color alone.

### Navigation
- Left navigation uses plain rows, clear selected state, and a separate visual group for volunteer history. Collapsed navigation must preserve understandable tooltips.

### Graph and Trail Nodes
- Running nodes use amber, completed nodes use green, error nodes use red, idle nodes use dashed neutral borders.
- Node labels must fit inside their boxes. Long technical names should be translated to concise human labels.

## 6. Do's and Don'ts

### Do:
- **Do** use the same status vocabulary in chat cards, Node Trail, Graph View, and logs.
- **Do** keep the center chat column calm and readable, with a maximum comfortable line length.
- **Do** make empty states useful with short capability prompts.
- **Do** preserve volunteer route, storage keys, and request behavior while sharing the visual language.
- **Do** verify every visible Chinese string before shipping.

### Don't:
- **Don't** use purple AI gradients, decorative glassmorphism, or fake dashboard previews.
- **Don't** mix icon families or hand-roll SVG icons.
- **Don't** use color alone to communicate error, success, or waiting states.
- **Don't** allow mojibake text, broken placeholders, or malformed Chinese punctuation in visible UI.
- **Don't** change backend SSE contracts, route names, or volunteer business logic during UI polish.
