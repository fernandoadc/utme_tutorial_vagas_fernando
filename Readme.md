# UTME for Social Media Job Analysis

In the ever-evolving landscape of online communication, the rise of hate speech has become a pressing concern. To face this challenge, UTME emerges as a powerful solution. This approach can also be replicated for other problems that the current technological scenario requires, as well as the analysis of the Social Media market, which companies are increasingly using to verify the best profiles of candidates for vacancies in this sector. This tutorial explores the motivation behind using the UTME and the importance of automated mining and monitoring of job advertisement texts.

### Motivation for Using UTME

Traditional methods of text classification often face limitations in adaptability across diverse taxonomies due to the requirement for extensive labeled datasets. UTME provides a versatile and unsupervised approach, allowing users to classify documents within a custom hierarchical taxonomy without the need for labeled data. The unsupervised taxonomy expansion feature further empowers users to dynamically generate subcategories based on document content, offering a granular understanding of the data.

### Step-by-Step Tutorial: Putting UTME into Action

Now, let's go step by step on how to take advantage of the UTME to identify the most in-demand job profiles on social media:

1. **Initialize UTME:**
   - Set up your UTME environment. UTME leverages Large Language Models (LLMs), demonstrating its effectiveness across proprietary, open-source, and low-computational-cost models. Its adaptability makes it a practical choice for users and organizations seeking advanced text mining capabilities without requiring extensive computational resources.


```python
from utme.UTME import UTME
from utme.BinaryClassifier import BinaryClassifier
from utme.TaxonomyClassifier import TaxonomyClassifier
from utme.SubcategoryGenerator import SubcategoryGenerator

# Initialize UTME with LLM (Language Model) credentials
llm_key = "your_llm_key"
llm_endpoint = "your_llm_url_endpoint"
llm_options = {'model': "openchat_3.5", 'max_tokens': 4096}
utme_base = UTME(llm_endpoint, llm_key, llm_options)
```


2. **Define Context and Taxonomy:**
   - Clearly defines the relevant context and taxonomy for identifying job vacancies. This step establishes the framework for classification.


```python
# Define context and taxonomy for job vacancies analysis
context = '''The main topics from the text are: Social Media Manager, Social Media Specialist, Social Media Marketing, Social Media Coordinator, Social Media Editor, Social Media Strategist, Social Media Producer, Social Media Intern, Social Media Campaign Management, Social Media Content Producer, Social Media Management, Social Media Analyst, Social Media Executive, Social Media Curator, Social Media Guru, Social Media Officer, Social Media Associate, Social Media Marketing Manager, Social Media Buyer, Social Media and Content Manager, Social Media and Content Producer, Social Media Marketing Specialist, Social Media Marketing Manager, Social Media Sales Representative, Social Media Producer, Social Media Content Creation and Promotion, Social Media Editor, Social Media Manager, Social Media Lead, Social Media Analyst, Social Media Assistant/PA, Social Media Marketing Intern, Social Media and Marketing Administrator, Social Media Coordinator, Social Media Specialist, Social Media Marketing, Social Media Strategist, Social Media Support Coordinator, Social Media Marketing Strategist, Social Media Jobs, Social Media and Content Specialist, Social Media Internship.'''

taxonomy = '''0 NONE
1 Social Media Manager
2 Social Media Specialist
3 Marketing & Social Media Specialist
4 Social Media Editor
5 Social Media Strategist/Producer
6 Marketing Executive
7 Business Development
8 Lead the creative strategy for key social accounts
9 Connect with us on Twitter and Instagram
10 Social Media Producer
11 Public Relations and Social Media Assistant Manager
12 Equal Employment Opportunity for Individuals with Disabilities
13 Social Media Content Curator and Manager
14 Social Media Marketing Manager
15 Job Opportunity: Client Services Agent (m/f/d) for ARABIC Social Media
16 Senior Business Reporter (financial/business/technology) – digital media / social media platform
17 Social Media Specialist
18 Content Associate for Social Media and Entrepreneurial Platform
19 Development and Social Media Coordinator
20 Social Media Manager II
21 Connect with us on Twitter and Instagram
22 Intern- Social Media
23 Social Media Curator
24 Social Media Marketing Manager for Showtime Networks
25 Social Media and Event Marketing Management'''
```

In the UTME framework, the definition of context and taxonomy is flexible, requiring a degree of experimentation and creativity. It can be tailored to the specifics of the Large Language Model (LLM) used by UTME. While context is crucial throughout the process, it plays a pivotal role in BinaryClassification, thereby filtering texts of interest before taxonomy mapping.

For the taxonomy, used in TaxonomyClassifier, there is a specific format to follow. Each taxonomy item is a line starting with an increasing numerical identifier. It is imperative that the initial taxonomy item is "0 NONE", serving as the second filter for texts of interest.

3. **Binary Classification for Text Filtering:**
   - Use the Binary Classifier to filter texts of interest, focusing on documents that may contain job profiles.

```python
# Start BinaryClassifier to filter documents of interest
bc = BinaryClassifier(utme_base, context)
y_pred = bc.classify(df.text.to_list())
df['y_pred'] = y_pred
df_filtered = df[df.y_pred == 'YES']
```

The BinaryClassifier module in UTME employs a few-shot prompt learning approach, utilizing only the provided context to make predictions.

4. **Taxonomy Mapping – First Level:**
   - Employ the Taxonomy Classifier to map documents within the predefined hierarchical taxonomy at the first level.

```python
# Start TaxonomyClassifier to map documents to predefined categories (First Level)
tc = TaxonomyClassifier(bc, taxonomy)
taxonomy_pred = tc.classify(df_filtered.text.to_list())
df_filtered['level1'] = taxonomy_pred
df_filtered_level1 = df_filtered[~df_filtered.level1.str.contains('NONE')]
```

The TaxonomyClassifier utilizes a user-defined taxonomy. By employing this module, users can effectively categorize and identify instances of the most in-demand jobs in their text data.

5. **Unsupervised Taxonomy Expansion – Second Level:**
   - Utilize the Subcategory Generator to dynamically expand the taxonomy, generating subcategories for more detailed analysis.

```python
# Perform Unsupervised Taxonomy Expansion (Second Level)
L = []
for category in df_filtered_level1.level1.unique():
    sg = SubcategoryGenerator(tc)
    df_category = df_filtered_level1[df_filtered_level1.level1 == category]

    # Generate subcategories for the selected category
    subcategories = sg.generate_subcategories(category, df_category.sample(expansion_sample_size, replace=True).text.to_list())

    # Classify documents using the expanded subcategories
    tc2 = TaxonomyClassifier(bc, subcategories)
    taxonomy_pred2 = tc2.classify(df_category.text.to_list())

    df_category['level2'] = taxonomy_pred2
    L.append(df_category)
df_filtered_level2 = pd.concat(L)
```

The SubcategoryGenerator in UTME is essential for unsupervised taxonomy expansion by creating subcategories within predefined categories. For example, if the TaxonomyClassifier identifies a document under "Social Media Strategy", the SubcategoryGenerator can generate subcategories such as "Social Media Profile Management" based on the content of the document. This unsupervised approach allows the system to discover subtopics without labeled data, improving exploratory analysis and taxonomy development.

Access df_filtered_level2 to see the mapping result for each document.

6. **Graph-Based Analysis for Exploration:**
   - Leverage UTME's graph-based analysis to visually explore document relationships, helping to identify and monitor vacancy profile patterns.


```python
# Start SubcategoryGenerator for graph generation
sg = SubcategoryGenerator(tc)
sg.graph_generation(df_filtered_level2)
sg.graph_export_cosmograph(df_filtered_level2)
``` 
![Graph - Hate Speech Analysis](https://github.com/fernandoadc/utme_tutorial_vagas_fernando/blob/main/fig_topics_graph.png "Graph - Job Vacancy Analysis")

The UTME also facilitates graph analysis through the generated nodes.csv and edges.csv files, allowing for exploratory analysis of the results. In this graph, each hate speech text serves as a vertex, and similar texts are connected. UTME generates connections by exploring both document similarity and the predefined hate speech taxonomy. For analyzing large graphs, the Cosmograph app is recommended, providing robust features for graph visualization and exploration.

To navigate the taxonomy, we also suggest using TreeMaps:

```python
# Treemap
import plotly.express as px
df_tree = utme_hatespeech.df_filtered_level2[['text','level1','level2']]
fig = px.treemap(df_tree, path=['level1', 'level2'],  color='level2',  color_continuous_scale='RdBu')
fig.show()
``` 
![Graph - Hate Speech Analysis](https://github.com/fernandoadc/utme_tutorial_vagas_fernando/blob/main/img2.png)"TreeMap - Job Vacancyes Analysis")

# UTME_HateSpeech Class

The UTME_HateSpeech class consolidates all the stages of UTME into a single, user-friendly interface, simplifying the hate speech analysis process for users and analysts. This class encapsulates the Binary Classifier for text filtering, the Taxonomy Classifier for mapping documents to a predefined taxonomy, and the Unsupervised Taxonomy Expansion through the Subcategory Generator.

```python
# UTME Hate Speech Analysis Tutorial

from utme.UTME import UTME
from utme.BinaryClassifier import BinaryClassifier
from utme.TaxonomyClassifier import TaxonomyClassifier
from utme.SubcategoryGenerator import SubcategoryGenerator
import pandas as pd

class UTME_HateSpeech():

    def __init__(self, utme_base):
        self.utme_base = utme_base

    def start(self, df, context, taxonomy, expansion_sample_size=10):
        """
        Start the UTME_HateSpeech process for classifying hate speech in text data.

        Parameters:
        - df (pd.DataFrame): DataFrame containing text data to be processed.
        - context (str): Example of domain-application interest documents.
        - taxonomy (list): List of predefined categories for taxonomy classification.
        - expansion_sample_size (int): Size of the sample for taxonomy expansion (default is 10).
        """
        print('# Binary Classifier for Text Filtering')
        bc = BinaryClassifier(self.utme_base, context)
        y_pred = bc.classify(df.text.to_list())
        df['y_pred'] = y_pred
        df_filtered = df[df.y_pred == 'YES']
        print('Selected', len(df_filtered), 'documents of', len(df))
        self.df_filtered = df_filtered

        print('# Taxonomy Document Mapping - First Level')
        tc = TaxonomyClassifier(bc, taxonomy)
        taxonomy_pred = tc.classify(df_filtered.text.to_list())
        df_filtered['level1'] = taxonomy_pred
        df_filtered_level1 = df_filtered[~df_filtered.level1.str.contains('NONE')]
        print('Selected', len(df_filtered_level1), 'documents of', len(df_filtered))
        self.df_filtered_level1 = df_filtered_level1

        print('# Unsupervised Taxonomy Expansion - Second Level')
        L = []
        for category in df_filtered_level1.level1.unique():
            print('Expanding:', category)
            sg = SubcategoryGenerator(tc)
            df_category = df_filtered_level1[df_filtered_level1.level1 == category]
            
            # Generate subcategories for the selected category
            subcategories = sg.generate_subcategories(category, df_category.sample(expansion_sample_size, replace=True).text.to_list())

            # Classify documents using the expanded subcategories
            tc2 = TaxonomyClassifier(bc, subcategories)
            taxonomy_pred2 = tc2.classify(df_category.text.to_list())

            df_category['level2'] = taxonomy_pred2
            L.append(df_category)
        df_filtered_level2 = pd.concat(L)
        self.df_filtered_level2 = df_filtered_level2

        print('# Graph generation for exploratory analysis')
        sg = SubcategoryGenerator(tc)
        sg.graph_generation(df_filtered_level2)
        sg.graph_export_cosmograph(df_filtered_level2)

        print('# UTME completed')

# Usage Example
llm_key = "your_llm_key"
llm_endpoint = "your_llm_url_endpoint"
llm_options = {'model': "openchat_3.5", 'max_tokens': 1024}
utme_base = UTME(llm_endpoint, llm_key, llm_options)
utme_hatespeech = UTME_HateSpeech(utme_base)


# Dataset
!wget https://huggingface.co/api/datasets/hate_speech18/parquet/default/train/0.parquet
df_hatespeech = pd.read_parquet('0.parquet')

df_label_yes = df_hatespeech[df_hatespeech.label==1].sample(250)
df_label_yes['y_true'] = 'YES'

df_label_no = df_hatespeech[df_hatespeech.label==0].sample(250)
df_label_no['y_true'] = 'NO'

df = pd.concat([df_label_yes,df_label_no])

# Define context and taxonomy for hate speech analysis
context = '''Hate speech refers to any form of communication that promotes prejudice, discrimination, or animosity against individuals or groups based on attributes such as race, ethnicity, religion, gender, sexual orientation, or other defining characteristics. Racial: Discriminatory language targeting a person's race or ethnicity. Religious: Prejudice or hostility based on someone's religious beliefs. Gender-based: Discrimination based on gender, often reinforcing traditional stereotypes. Sexual Orientation: Derogatory remarks about someone's sexual orientation. Disability-based: Offensive language related to a person's physical or mental abilities. Age-based: Discrimination based on a person's age, often manifesting as ageism. Nationality-based: Prejudice against individuals from a specific country or nationality. Class-based: Discrimination based on socioeconomic status or class. Appearance-based: Offensive remarks about a person's physical appearance. Language-based: Discriminatory language targeting a specific language or dialect. Political Affiliation: Prejudice based on a person's political beliefs or affiliation. Ideology-based: Discrimination related to a person's ideological views. Immigration Status: Discrimination based on a person's immigration or citizenship status. Body-shaming: Derogatory comments about a person's body size or shape. Educational Background: Discrimination based on a person's educational achievements or lack thereof. Refugees: Offensive language targeting individuals who are refugees or seeking asylum. Geographical Location: Discrimination based on a person's place of origin or residence. Weight-based: Offensive comments about a person's weight. Sexual Harassment Language: Offensive comments of a sexual nature that contribute to a hostile environment. Microaggressions: Subtle, often unintentional, expressions of prejudice or discrimination.'''

taxonomy = '''0 NONE: The @DOCUMENT does not contain hate speech
1 Racial Hate Speech
2 Religious Hate Speech
3 Gender-based Hate Speech
4 Sexual Orientation Hate Speech
5 Disability-based Hate Speech
6 Class-based Hate Speech
7 Political Hate Speech
8 Age-based Hate Speech
9 Appearance-based Hate Speech
10 Nationality-based Hate Speech'''

utme_hatespeech.start(df, context, taxonomy)
```


## UTME HateSpeech in Action using Google Colab

### Google Colab 1: LLM Server

To get a hands-on experience with the UTME HateSpeech functionality and test its concepts, you can access the  Google Colab notebook [here](https://colab.research.google.com/drive/1mvLCbjmTFU_Fn3YLgGarLEhN_pks2yNj?usp=sharing). This notebook provides a simple Large Language Model (LLM) server setup to explore the basic capabilities of the UTME, allowing you to understand the Binary Classifier, Taxonomy Classifier, and Subcategory Generator. This serves as a preliminary step to familiarize yourself with the UTME HateSpeech workflow.

### Google Colab 2: UTME HateSpeech Implementation and Experiment

For a more comprehensive demonstration, the second Google Colab notebook [here](https://colab.research.google.com/drive/1O4jBqTHyHUSkslCWsAF5QvCz3awAt-Tj?usp=sharing) showcases the full implementation of the UTME_HateSpeech class. Additionally, a small experiment using hate speech texts available in the [Hugging Face Hate Speech dataset](https://huggingface.co/api/datasets/hate_speech18/) is included. This notebook guides you through applying UTME HateSpeech on real hate speech data, demonstrating the effectiveness of the solution in filtering, classifying, and expanding the taxonomy for a more detailed hate speech analysis.

Feel free to explore these notebooks to gain practical insights into the UTME HateSpeech capabilities and witness its application in addressing hate speech challenges within textual datasets.
