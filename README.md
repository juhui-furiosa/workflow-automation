# AI Workflow Automator

This project is a LangGraph-based workflow automator that uses natural language commands to summarize calendar events, send messages to Slack, and log updates to Notion.

Example command:
> Summarize today’s schedule and send it to Slack and Notion.

## Prerequisites
If you have access to the FurisoaAI LLM API endpoint or a dedicated RNGD server, the only step you need to take before beginning the SDK cookbook is installing the necessary prerequisites for the FuriosaAI SDK stack. Please refer to the provided page to choose and set up your preferred version of the FuriosaAI SDK. Currently, this SDK cookbook is based on FuriosaAI SDK version 2025.3.0.
- [Install Prerequisites for FuriosaAI SDK](https://developer.furiosa.ai/latest/en/get_started/prerequisites.html)

## Set Environments 

### Requirements
- FuriosaSDK 2025.3.0.
- RNGD server or Endpoint API

### Installation
```
   git clone https://github.com/furiosa-ai/sdk-cookbook.git
   cd workflow-automation
   python -m venv venv
   source /venv/bin/activate
   pip install -r requirements.txt
```

### Set Environment Variables
Create a `.env` file in the project root based on `.env.example`:
```
   cp .env.example .env
   vi .env
```
Then, fill in your credentials:
- `SLACK_WEBHOOK_URL`: Slack webhook for posting summaries
- `NOTION_TOKEN`: Notion API integration token
- `NOTION_PAGE_ID`: Page ID to write summaries
- `NOTION_DB_ID`: Database ID to write summaries
- `ICAL_URL`: URL to an `.ics` calendar file

## Configuration
- LLM:
   - [furiosa-ai/Llama-3.1-8B-Instruct](https://huggingface.co/furiosa-ai/Llama-3.1-8B-Instruct) 

## Language Support
This AI Workflow Automator supports both Korean and English languages:
- 🇰🇷 Korean: 오늘 일정 요약해서 슬랙에 보내고 노션에도 기록해
- 🇺🇸 English: Summarize today’s schedule and send it to Slack and Notion.

You can interact with the system using either language for your workflow automation needs.

## Usage

Execute the `.py` file using the command below to test the recipe for building an AI Workflow Automator that connects calendar data with Slack and Notion through natural language commands.  
This project applies the concept of workflow orchestration using LangGraph to automatically summarize, post, and log daily schedules.

You can enter any natural language instruction (e.g., “Summarize today’s schedule and send it to Slack and Notion”), and the workflow will automatically plan, fetch, summarize, and execute the required actions.

```
python main.py
```

## File Structure

```
workflow_automation/
│
├── main.py                  
├── requirements.txt          
├── .env                     
└── README.md
