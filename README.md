# Finance_Chatbot
<p align="center"><img width="1000" height="300" src="https://surveysparrow.com/wp-content/uploads/2020/11/Advantages-of-adding-chatbot-for-website-@2x-Copy-1.png"></p>

This project initiated from **ChatBot**, using advance language model like GPT2 and advance words embedding to build a chatbot for banking service, focusing on answering questions raised by customers for banking services, with the big plan on creating a delightful shopping experience. With this conversation data - <code>intents.json</code> with <font color='blue'>tags as target, where patterns as questions</font>, we would create the **ChatBot** that would tackle customers need on any finance inquiries.

## Table of Contents
* **1. About the Project**
* **2. Getting Started**
* **3. Set up your environment**
* **4. Open your Jupyter notebook**


## Structuring a repository
An integral part of having reusable code is having a sensible repository structure. That is, which files do we have and how do we organise them.
- Folder layout:
```bash
customer_segmentation
├── src
│   └── static
│       └── images
│         └── chatbox-icon.svg
│       └── app.js
|       └── style.css
│   └── templates
│       └── index.html
│   └── preprocess
│       └── __init__.py
|       └── preprocess.py
|       └── nltk_utls.py
|       └── parse_data.py
|   └── train
│       └── __init__.py
│       └── feedforward.py
│       └── dialog_gpt2.py
|       └── models.py
|   └── config.py
|   └── app.py
|   └── chat.py
├── .gitignore
├── README.md
├── requirements.txt
├── Attention_is_All_You_Need.ipynb
└── DialogGPT_Rakchat.ipynb
```


## 1. About the Project
With this conversational data - <font color='blue'>intents.json</font>, let kick started on it:
  - <b><u>Creating finance conversation data</u></b>
  - <b><u>Preprocess the text information given in the conversation data like stemming, removing stopwords, lematization...</u></b>
  - <b><u>Word embedding, creating words vectors using techniques like Bags of Words (BOW)</u></b>
  - <b><u>Creating language model like transformers and GPT for chatbot</u></b>
  - <b><u>Traing the language for conversations</u></b>
  

## 2. Getting Started
- Prefer to use the `conda` package manager (which ships with the Anaconda distribution of Python),
  1. Clone the repository locally
    In your terminal, use `git` to clone the repository locally.
    
    ```bash
    https://github.com/KewJS/Finance_Chatbot.git
    ```
    
    Alternatively, you can download the zip file of the repository at the top of the main page of the repository. 
    If you prefer not to use git or don't have experience with it, this a good option.
    
- Prefer to use `pipenv`, which is a package authored by Kenneth Reitz for package management with `pip` and `virtualenv`, or


## 3. Set up your environment

### 3a. `conda` users

If this is the first time you're setting up your compute environment, 
use the `conda` package manager 
to **install all the necessary packages** 
from the provided `environment.yml` file.

```bash
conda env create -f environment.yml
```

To **activate the environment**, use the `conda activate` command.

```bash
conda activate customer_segmentation
```

**If you get an error activating the environment**, use the older `source activate` command.

```bash
source activate customer_segmentation
```

To **update the environment** based on the `environment.yml` specification file, use the `conda update` command.

```bash
conda env update -f environment.yml
```

### 3b. `pip` users

Please install all of the packages listed in the `requirement.txt`. 
An example command would be:

```bash
pip install -r requirement.txt
```


## 4. Open your Jupyter notebook

1. You will have to install a new IPython kernelspec if you created a new conda environment with `environment.yml`.
    
    ```python
    python -m ipykernel install --user --name customer_segmentation --display-name "customer_segmentation"
    ```

You can change the `--display-name` to anything you want, though if you leave it out, the kernel's display name will default to the value passed to the `--name` flag.

2. In the terminal, execute `jupyter notebook`.

Navigate to the notebooks directory and open notebook:
  - Preprocessing: `Preprocess.ipynb`
  - Modelling: `Train.ipynb`
