import argparse
import base64
import os
import re

def get_base64_img(img_path):
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        return ""
    ext = os.path.splitext(img_path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/{ext};base64,{encoded_string}"

def main():
    parser = argparse.ArgumentParser(description="Convert a markdown canopy report into a standalone HTML file.")
    parser.add_argument("--repo-dir", default="/Users/cherrytian/Documents/GitHub/embeddings_esri")
    parser.add_argument("--md-path", default=None)
    parser.add_argument("--html-out", default=None)
    parser.add_argument("--title", default="London Trees Outside Woodland: Advanced Canopy Health Analysis")
    args = parser.parse_args()

    repo_dir = args.repo_dir
    md_path = args.md_path or os.path.join(repo_dir, "tree_health_report.md")
    html_out = args.html_out or os.path.join(repo_dir, "tree_health_report_standalone.html")
    
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    import json
    # Find and collect base64 data for all images to embed them separately from the markdown source
    # Format: ![alt](path)
    img_regex = r"!\[([^\]]*)\]\(([^)]+)\)"
    images_dict = {}
    
    for match in re.finditer(img_regex, md_content):
        path = match.group(2)
        full_path = os.path.join(repo_dir, path)
        base64_data = get_base64_img(full_path)
        if base64_data:
            images_dict[path] = base64_data
            
    images_json = json.dumps(images_dict)

    # Let's create the standalone HTML template
    html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>###TITLE###</title>
    
    <!-- Marked for Markdown Parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    
    <!-- KaTeX for Math Equations -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js"></script>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --text-dark: #1f2937;
            --text-muted: #4b5563;
            --bg-body: #f9fafb;
            --bg-card: #ffffff;
            --border: #e5e7eb;
            --radius-lg: 12px;
            --radius-md: 8px;
            --font-sans: 'Inter', sans-serif;
            --font-title: 'Outfit', sans-serif;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-sans);
            color: var(--text-dark);
            background-color: var(--bg-body);
            line-height: 1.6;
            padding: 0;
            display: flex;
            min-height: 100vh;
        }

        /* Sidebar navigation */
        .sidebar {
            width: 300px;
            background: #0f172a;
            color: #f8fafc;
            padding: 2.5rem 1.5rem;
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            overflow-y: auto;
            border-right: 1px solid rgba(255,255,255,0.05);
            z-index: 10;
        }

        .sidebar h2 {
            font-family: var(--font-title);
            font-size: 1.25rem;
            margin-bottom: 2rem;
            color: #38bdf8;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .sidebar ul {
            list-style: none;
        }

        .sidebar li {
            margin-bottom: 0.75rem;
        }

        .sidebar a {
            color: #94a3b8;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.2s ease;
            display: block;
            padding: 0.35rem 0.5rem;
            border-radius: var(--radius-md);
        }

        .sidebar a:hover, .sidebar a.active {
            color: #ffffff;
            background: rgba(255,255,255,0.08);
            transform: translateX(4px);
        }

        /* Main content area */
        .content-container {
            margin-left: 300px;
            padding: 3rem 4rem;
            max-width: 1000px;
            width: 100%;
            background: var(--bg-body);
        }

        .report-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 3.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        }

        /* Typography & elements */
        h1, h2, h3, h4 {
            font-family: var(--font-title);
            color: #0f172a;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
            letter-spacing: -0.025em;
        }

        h1 {
            font-size: 2.25rem;
            line-height: 1.25;
            margin-top: 0;
            margin-bottom: 2rem;
            color: var(--primary);
            border-bottom: 2px solid var(--border);
            padding-bottom: 1.5rem;
        }

        h2 {
            font-size: 1.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
            margin-top: 3rem;
        }

        h3 {
            font-size: 1.2rem;
            margin-top: 2rem;
        }

        p {
            margin-bottom: 1.25rem;
            font-size: 1.05rem;
            color: #374151;
        }

        strong {
            font-weight: 600;
            color: #111827;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
            font-size: 0.95rem;
            text-align: left;
            border-radius: var(--radius-md);
            overflow: hidden;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border);
        }

        th {
            background-color: #f8fafc;
            color: #0f172a;
            font-weight: 600;
            padding: 0.75rem 1rem;
            border-bottom: 2px solid var(--border);
        }

        td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            color: #334155;
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:nth-child(even) {
            background-color: #f8fafc;
        }

        /* Images */
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 2rem auto;
            border-radius: var(--radius-md);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border);
        }

        /* Callouts / Alerts */
        .callout {
            padding: 1.25rem 1.5rem;
            margin: 1.5rem 0;
            border-radius: var(--radius-md);
            border-left: 4px solid;
            font-size: 0.975rem;
        }

        .callout-important {
            background-color: #eff6ff;
            border-color: #3b82f6;
            color: #1e3a8a;
        }
        
        .callout-tip {
            background-color: #ecfdf5;
            border-color: #10b981;
            color: #065f46;
        }

        .callout-title {
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Code blocks */
        pre {
            background-color: #f1f5f9;
            padding: 1.25rem;
            border-radius: var(--radius-md);
            overflow-x: auto;
            margin: 1.5rem 0;
            border: 1px solid var(--border);
        }

        code {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            background-color: #f1f5f9;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            color: #0f172a;
        }

        pre code {
            padding: 0;
            background-color: transparent;
            color: inherit;
        }

        hr {
            border: 0;
            height: 1px;
            background: var(--border);
            margin: 3rem 0;
        }

        /* Math alignment */
        .katex-display {
            margin: 1.5rem 0 !important;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 0.5rem 0;
        }

        /* Floating print action button */
        .print-btn {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 50px;
            font-family: var(--font-title);
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.3);
            transition: all 0.2s ease;
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .print-btn:hover {
            background: var(--primary-light);
            transform: translateY(-2px);
            box-shadow: 0 12px 20px -3px rgba(59, 130, 246, 0.4);
        }

        /* Print styles */
        @media print {
            body {
                background: white !important;
                color: black !important;
            }

            .sidebar, .print-btn {
                display: none !important;
            }

            .content-container {
                margin-left: 0 !important;
                padding: 0 !important;
                max-width: 100% !important;
                width: 100% !important;
            }

            .report-card {
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
            }

            h2, h3 {
                page-break-after: avoid;
            }

            tr, img, .callout {
                page-break-inside: avoid;
            }

            a {
                text-decoration: underline;
                color: black !important;
            }
        }

        @media (max-width: 768px) {
            body {
                flex-direction: column;
            }
            .sidebar {
                width: 100%;
                position: static;
                padding: 1.5rem;
            }
            .content-container {
                margin-left: 0;
                padding: 1.5rem;
            }
            .report-card {
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>

    <div class="sidebar">
        <h2>Report Sections</h2>
        <ul id="toc">
            <!-- Dynamically populated -->
        </ul>
    </div>

    <div class="content-container">
        <article class="report-card" id="output">
            <!-- Markdown parsed content will go here -->
        </article>
    </div>

    <button class="print-btn" onclick="window.print()">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 6 2 18 2 18 9"></polyline><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path><rect x="6" y="14" width="12" height="8"></rect></svg>
        Save to PDF / Print
    </button>

    <!-- Markdown text container -->
    <script type="text/markdown" id="markdown-source">###CONTENT###</script>

    <script>
        const EMBEDDED_IMAGES = ###IMAGES_JSON###;
        (function() {
            const outputDiv = document.getElementById("output");
            try {
                if (typeof marked === 'undefined') {
                    throw new Error("Marked.js library failed to load. Please check your internet connection or CDN availability.");
                }
                const rawMarkdown = document.getElementById("markdown-source").textContent;
                
                // Custom parser for math equations before marked parses it
                // We temporarily replace equations so marked doesn't touch them
                const mathBlocks = [];
                let processed = rawMarkdown;
                
                // Display math $$ ... $$
                processed = processed.replace(/\$\$([\s\S]+?)\$\$/g, (match, math) => {
                    const id = `__MATH_BLOCK_${mathBlocks.length}__`;
                    mathBlocks.push({ id, math, display: true });
                    return id;
                });
                
                // Inline math $ ... $
                processed = processed.replace(/\$([^$]+?)\$/g, (match, math) => {
                    const id = `__MATH_BLOCK_${mathBlocks.length}__`;
                    mathBlocks.push({ id, math, display: false });
                    return id;
                });

                // Handle GitHub alerts custom formatting
                // Examples:
                // > [!IMPORTANT]
                // > content
                processed = processed.replace(/^>\s+\[!(IMPORTANT|TIP|WARNING|CAUTION|NOTE)\]\s*\n((?:>\s+.*\n?)*)/gm, (match, type, content) => {
                    const cleanContent = content.replace(/^>\s+/gm, '');
                    return `<div class="callout callout-${type.toLowerCase()}">` +
                           `<div class="callout-title">${type}</div>` +
                           `<div>${cleanContent}</div>` +
                           `</div>\n`;
                });

                // Parse Markdown
                let htmlContent = marked.parse(processed);

                // Re-insert math rendered with KaTeX
                if (typeof katex !== 'undefined') {
                    mathBlocks.forEach(({ id, math, display }) => {
                        try {
                            const rendered = katex.renderToString(math, {
                                displayMode: display,
                                throwOnError: false
                            });
                            htmlContent = htmlContent.replace(id, rendered);
                        } catch (e) {
                            console.error("KaTeX error:", e);
                            htmlContent = htmlContent.replace(id, math);
                        }
                    });
                } else {
                    console.warn("KaTeX is not loaded, rendering raw math blocks.");
                    mathBlocks.forEach(({ id, math }) => {
                        htmlContent = htmlContent.replace(id, math);
                    });
                }

                outputDiv.innerHTML = htmlContent;

                // Replace image src attributes with embedded base64 data
                const images = outputDiv.querySelectorAll("img");
                images.forEach(img => {
                    const src = img.getAttribute("src");
                    if (src && EMBEDDED_IMAGES[src]) {
                        img.src = EMBEDDED_IMAGES[src];
                    }
                });

                // Generate Table of Contents
                const toc = document.getElementById("toc");
                const headers = document.querySelectorAll("#output h2, #output h1");
                headers.forEach((h, index) => {
                    // Ensure heading has an ID
                    if (!h.id) {
                        h.id = "heading-" + index;
                    }
                    const li = document.createElement("li");
                    const a = document.createElement("a");
                    a.href = "#" + h.id;
                    a.textContent = h.textContent;
                    
                    // Indent h2 items slightly
                    if (h.tagName === "H2") {
                        li.style.marginLeft = "0.75rem";
                    }
                    
                    li.appendChild(a);
                    toc.appendChild(li);
                });

                // Scroll spy for active sidebar link
                const links = toc.querySelectorAll("a");
                window.addEventListener("scroll", () => {
                    let current = "";
                    headers.forEach(h => {
                        const top = window.scrollY;
                        const offset = h.offsetTop - 120;
                        if (top >= offset) {
                            current = h.id;
                        }
                    });

                    links.forEach(a => {
                        a.classList.remove("active");
                        if (a.getAttribute("href") === "#" + current) {
                            a.classList.add("active");
                        }
                    });
                });
            } catch (err) {
                console.error(err);
                outputDiv.innerHTML = `<div style="padding: 2rem; color: #b91c1c; background: #fef2f2; border: 1px solid #fee2e2; border-radius: 8px;">
                    <h3 style="margin-top: 0; color: #b91c1c;">Report Rendering Error</h3>
                    <p style="font-weight: bold;">${err.message}</p>
                    <pre style="background: #cbd5e1; padding: 1rem; border-radius: 4px; overflow-x: auto; font-family: monospace; font-size: 0.85rem;">${err.stack}</pre>
                </div>`;
            }
        })();
    </script>
</body>
</html>"""

    # Insert clean markdown and the image JSON lookup into the template
    final_html = html_template.replace("###TITLE###", args.title)
    final_html = final_html.replace("###CONTENT###", md_content)
    final_html = final_html.replace("###IMAGES_JSON###", images_json)

    with open(html_out, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"Standalone HTML report successfully written to {html_out}")

if __name__ == "__main__":
    main()
