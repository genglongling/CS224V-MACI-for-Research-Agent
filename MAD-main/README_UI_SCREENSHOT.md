# Converting HTML UI to LaTeX Figure

## Option 1: Manual Screenshot (Easiest)

1. Open the HTML file in your browser:
   ```bash
   open "/Users/glin/Desktop/cs224v/CollectiveMind Interactive Debate.html"
   ```

2. Take a screenshot:
   - **macOS**: Press `Cmd + Shift + 4`, then drag to select the area
   - **Windows**: Use Snipping Tool or `Win + Shift + S`
   - **Linux**: Use `gnome-screenshot` or similar

3. Save the screenshot as `ui_screenshot.png` in the `MAD-main/` directory

4. The LaTeX code is already added to `paper_updated.tex` - it will automatically include the image!

## Option 2: Automated Conversion (Requires Playwright)

If you want to automate this:

1. Install playwright:
   ```bash
   pip install playwright
   playwright install chromium
   ```

2. Run the conversion script:
   ```bash
   cd MAD-main
   python convert_html_to_image.py "/Users/glin/Desktop/cs224v/CollectiveMind Interactive Debate.html" ui_screenshot.png
   ```

3. The image will be saved as `ui_screenshot.png` in the `MAD-main/` directory

## Option 3: Using Browser Developer Tools

1. Open the HTML file in Chrome/Edge
2. Press `F12` to open Developer Tools
3. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
4. Type "screenshot" and select "Capture full size screenshot"
5. Save as `ui_screenshot.png` in `MAD-main/` directory

## Notes

- The LaTeX code expects the image file to be named `ui_screenshot.png` and located in the same directory as `paper_updated.tex`
- The image will be scaled to fit the page width (`\textwidth`)
- If you want to adjust the size, modify the `width` parameter in the `\includegraphics` command

