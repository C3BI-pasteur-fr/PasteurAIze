<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=300px src="logo.png" alt="Project logo" style="border-radius: 10%;"></a>
</p>

<h3 align="center">PasteurAIze</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()


</div>

---

<p align="center"> Context-Engineered AI for Secure Clinical-Biological Data
Analytics
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [How to structure my data](#structuredData)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)

## 🧐 About <a name = "about"></a>

PasteurAIze uses modular AI agents and the Model Context Protocol to turn natural-language questions into reproducible biomedical analyses.

PasteurAIze will transform biomedical data analysis at Institut Pasteur through “context-engineered large-language-model (LLM) agents. It will converts a plain-language query (e.g. “Does age affect lymphocyte counts?”) into executable code, runs the code on
institutional resources and returns fully documented results.

PasteurAIze leverages the institutional LLM repository to carefully manage a full control over data governance and the model size for sobriety concerns. Its architecture features 3 core components: 
1) A text-to-SQL
agent converting natural language to database queries.
 2) A visualization agent using Vega-lite MCP
for graphic.
3) A scientific literature agent employing Google Scholar MCP with Retrieval-Augmented Generation (RAG).

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software and how to install them.
- Have UV installed:  
```
pip install uv
```
- Have an acess to LLM (base_url and api_key).
- Have a local database (ex: postgresql).
- Have some projects data well structured (to know how to structure your data go to [How to structure my data](#structuredData)).
- The file extensions handled for dataframes are : `.parquet`,`.tsv`,`.csv`.
- All data for one project must be in the same folder _(including yaml if provided)_. Set the `FOLDER_PATH`in your `.env`with the path to this folder.  
- _optional : yaml files describing your data (following the example.yaml schema in the pasteuraize folder)._
### Installing

A step by step series of examples that tell you how to get a development env running.

1) Create the .venv and install dependencies with :
```
uv sync
```
2) Install vanna with:

```
uv pip install vanna
```
3) Create a .env file and place in it each field from the .env.exemple file (fulfilled by yourself).

4) Add your data in the database and train Vanna by using the `auto_table.py`
⚠️ You can add only one project per execution and folder. If you want to add another project; change `FOLDER_PATH` and use again `auto_table.py`.

5) Give the name of the project that you are inserting and wait.


6) Once done; start chatting with :
```
streamlit run pasteuraize.py
```


## 📊 How to structure my data <a name="structuredData"></a>
A series of rules to follow to make your data structured as wanted by the application.

- Follow this usual project structure : one table/dataframe named `target`, one `taxonomy` and one `count_table`.

- `Primary keys` must be the first columns.
- `Foreign keys`must be named as `YourName_FK`.
- `Foreign keys`must have the same name as the primary keys from other tables (with the `_FK`).
- _optional : if you want to use an already existing yaml it should be named : `name_of_the_table.yaml`_ 


## 🎈 Usage <a name="usage"></a>

<div align="center">
  <img src="https://github.com/user-attachments/assets/f0853328-88af-442f-9b7e-63dfb2981511" alt="PasteurAIze Demo" width="700">
</div>








## ⛏️ Built Using <a name = "built_using"></a>

- [Pydantic-AI](https://ai.pydantic.dev) - Agent Framework
- [Lite-LLM](https://www.litellm.ai/#hero) - LLM Access
- [Streamlit](https://streamlit.io) - Web Framework
- [Pandas](https://pandas.pydata.org) - Dataframe Treatment
- [Crawl4AI](https://docs.crawl4ai.com) - Scraping & Crawling
- [Pydantic-Logfire](https://pydantic.dev/logfire) - Debugging Agent
- [Scholarly](https://github.com/scholarly-python-package/scholarly) - Searching papers on Google Scholar



