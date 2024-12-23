# Content Planning Flow

AI-powered content planning system using CrewAI Flows to transform blog posts into optimized social media content.

## Core Features

- 🔄 **Automated Blog Post Scraping**: Integrate with FireCrawl for efficient content extraction
- 🤖 **AI-Driven Content Analysis**: Intelligent parsing and understanding of blog content
- 📱 **Multi-Platform Support**: Optimize content for Twitter and LinkedIn
- 📊 **Automated Scheduling**: Smart content distribution timing
- 🔌 **Extensible Architecture**: Agent-based system for easy customization

## Prerequisites

Before you begin, ensure you have:

- Python 3.11 or higher
- OpenAI API key
- FireCrawl API key
- Typefully API access

## Installation Guide

1. Clone the repository:
```bash
git clone https://github.com/yourusername/content-planner-flow.git
cd content-planner-flow
```

2. Set up a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Unix or MacOS:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Configure your environment:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Quick Start

1. Launch Jupyter notebook:
```bash
jupyter notebook notebook.ipynb
```

2. Set your target blog URL in the notebook:
```python
blog_post_url = "https://your-blog-url.com"
```

3. Execute all cells to generate and schedule your content

## Project Structure

```
├── config/
│   ├── planner_agents.yaml    # Agent configurations
│   └── planner_tasks.yaml     # Task definitions
├── notebook.ipynb             # Main workflow
├── requirements.txt           # Dependencies
├── thread/                    # Generated content
└── workdir/                   # Temporary files
```

## Configuration

### Agent Configuration

Edit `config/planner_agents.yaml` to customize agent behaviors:

```yaml
draft_analyzer:
  role: "Content Analysis Expert"
  goal: "Analyze blog content..."

twitter_thread_planner:
  role: "Twitter Thread Expert"
  goal: "Create engaging threads..."
```

### Task Configuration

Modify `config/planner_tasks.yaml` to adjust task parameters:

```yaml
analyze_draft:
  description: "Analyze blog content..."
  
create_twitter_thread_plan:
  description: "Plan Twitter thread..."
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue in the GitHub repository.