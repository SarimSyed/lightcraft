# Photo navigation

Move the pointer over the photo and hold **Ctrl while scrolling** to zoom in or
out. Zoom follows the pointer, with keyboard steps from 6–1600% and continuous gestures bounded by Fit–1600%. Native pinch and two-finger pan also work. The filmstrip, grid and editing panels
keep their usual scrolling. Detail, full-screen preview, Compare and Reference
support photo zoom; Compare keeps the two views synchronized.

Drag the photo to pan. With an editing tool active, use **middle-button drag**.
Middle drag always moves the view without painting or changing a crop. The
Navigator also moves to another part of the photo.

**Settings → Navigation** changes the wheel modifier (Ctrl, Alt, Shift, Cmd, none,
or disabled), drag button, and an optional pan modifier used with editing tools.
The default leaves tool modifiers available for editing (for example Alt erases
with the masking brush). It also rebinds
Zoom In, Zoom Out, Fit, 100% and Toggle Zoom. Enter a shortcut such as `Alt+I` and
leave the field to apply it; an empty field disables that binding. Conflicts with
existing actions are rejected. Reset Navigation restores the defaults. App
preferences persist in `ui.json`, independently of photo edits and libraries.
Changing Toggle Zoom also replaces its secondary Space binding.

**Help → Photo Navigation** explains the current bindings and links to Settings.
The Keyboard Shortcuts sheet includes the same navigation help. On Linux and
Windows, menu `Cmd` shortcuts mean Ctrl; the wheel's Ctrl option means the physical
Control key on every platform.

## AI Denoise preview

The dialog starts with a 100% source crop. The crop navigator can choose any part
of the image; arrow keys move the crop. The configured wheel gesture zooms from
50–800%, and dragging pans. Moving waits briefly for the gesture to finish, then
prepares an asynchronous source-specific preview. The previous crop remains
visible during preparation. Apply is unavailable until the current crop is ready.
Hold Before or Space to compare the same area. The 100% button restores native
pixel scale. Amount changes blend the prepared correction without inference.

Navigation does not create edits. Cancel leaves saved settings alone. Apply still
processes the **whole source**, regardless of which crop is shown. Contributing
tiles remain anchored to the full-source grid, matching full-image inference.

The main Detail view keeps a canvas-sized whole-image preview for the Navigator/histogram and
adds a source-resolution window when zoomed in. The default Preview size cap applies to the
whole-image render; it does not cap the zoom window. Before/After uses matching windows. Compare,
overlays and soft proofing still have coverage limitations described in the parity tracker.
For denoise assessment, use native 100% in the AI dialog or Detail and compare with a full export.
Both shortcut editors share the saved keymap; legacy fork zoom bindings migrate when loaded.

## Control channel

The existing `ui.scroll` accepts `ctrl`, `alt`, `shift` and `cmd`; `ui.drag` and
`ui.dragWidget` accept `button: "middle"`. UI and control callers use these commands:

| Command | Parameters |
| --- | --- |
| `view.zoomAt` | finite `x`, `y`, `factor` (0.1–10); optional `photo`, `pixelsPerPoint`, `image` and `viewport` rectangles in screen points |
| `view.pan` | finite normalized `dx`, `dy` in −1…1 |
| `app.settings` | `tab: "navigation"` |
| `app.navigationBinding` | `command` (one of the five zoom commands), `shortcut`; or `reset: true` |
| `app.navigationHelp` | opens the current bindings and help |
| `dialog.denoise.navigate` | optional normalized `center: [x,y]` and `zoom` (50–800); requires an open dialog |

Headless regressions exercise pointer anchoring, rebinding, conflict rejection,
menus/help, persistence, tool-safe dragging and real-checkpoint crop navigation.
