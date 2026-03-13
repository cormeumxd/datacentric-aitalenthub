---
name: kompas_crawler_skill
description: "Web crawling and scraping tool with LLM-optimized output. Web crawler, web scraper, spider. DuckDuckGo search, site crawling, dynamic page scraping.
---

# Crawl4AI Skill - Web Crawler & Scraper

**Web Crawling | Web Scraping | LLM**

Free web crawler and scraper with LLM-optimized Markdown output.

## Core Features

- 🔍 **Web Search** - DuckDuckGo search
- 🕷️ **Web Crawling** - Site crawler, spider, sitemap
- 📝 **Web Scraping** - Smart scraper, data extraction
- 📄 **LLM-Optimized Output** - Fit Markdown, Save Token 80%
- ⚡ **Dynamic Page Scraping** - JavaScript

---

## Quick Start

### Installation

```bash
pip install crawl4ai-skill
```

### Web Search

```bash
# Search the web with DuckDuckGo
crawl4ai-skill search "python web scraping"
```

### Web Scraping

```bash
# Scrape a single web page
crawl4ai-skill crawl https://example.com  
```

### Web Crawling

```bash
# Crawl entire website / spider
crawl4ai-skill crawl-site https://docs.python.org --max-pages 50
```

---

## Use Cases

### Use Case 1: Web Crawler for Documentation

```bash
# Crawl documentation site with spider
crawl4ai-skill crawl-site https://docs.fastapi.com --max-pages 100
```

**Crawler Output:**
- ❌ Removed: navigation, sidebar, ads
- ✅ Kept: titles, main content, code blocks
- 📊 **Token: 50,000 → 10,000 (-80%)**

### Use Case 2: Search + Scrape

```bash
# Search and scrape top results
crawl4ai-skill search-and-crawl "Vue 3 best practices" --crawl-top 3
```

### Use Case 3: Dynamic Page Scraping

JavaScript-rendered pages (social platforms, news sites, etc.):

```bash
# Scrape JavaScript-heavy pages
crawl4ai-skill crawl https://xueqiu.com/S/BIDU --wait-until networkidle --delay 2
```

---

## Commands

| Command | Description |
|------|------|
| `search <query>` | Web search |
| `crawl <url>` | Web scraping - single page |
| `crawl-site <url>` | Web crawling - entire site |
| `search-and-crawl <query>` | Search + scrape combined |

### Common Options

```bash
# Web Search
--num-results 10          # Number of results

# Web Scraping
--format fit_markdown     # Output format
--output result.md        # Output file
--wait-until networkidle  # Wait strategy for dynamic pages
--delay 2                 # Additional wait time (seconds)
--wait-for ".selector"    # Wait for specific element

# Web Crawling
--max-pages 100          # Max pages to crawl
--max-depth 3            # Max crawl depth
```

---

## Output Formats

### fit_markdown (Recommended)

Smart extraction, save 80% tokens.

```bash
crawl4ai-skill crawl https://example.com --format fit_markdown
```

### raw_markdown

Preserve full structure.

```bash
crawl4ai-skill crawl https://example.com --format raw_markdown
```

---

## Why This Crawler?

✅ **Free Crawler** - No API key required, ready to use  
✅ **Smart Scraper** - Auto noise removal, extract core content  
✅ **Site Crawler** - Sitemap support, recursive crawling  
✅ **Dynamic Scraping** - JavaScript-rendered page support  
✅ **Search Integration** - DuckDuckGo search built-in 
```