# PageSageXpert: A Search Engine for Wikipedia Data

<br>
**🥇 Award for "Exceptional Project" by Rutgers CS Department**
<br>
<br>
For full project report and detailed explanation, please read the report below:<br>
https://docs.google.com/document/d/15nH_5_L9TQo82f4hSuKDSVl_tmNtUKao/edit<br>

## **Project Outline**

<img width="781" alt="pagesagexpert_outline" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/18c78201-22e8-4e5f-b4b1-eef0958174f3">

_Dataset_: https://www.kaggle.com/datasets/kenshoresearch/kensho-derived-wikimedia-data<br>
_Youtube link to Graph City_: https://www.youtube.com/watch?v=azRk8F7-KRE<br>
_Original Graph City Repository_: https://github.com/endlesstory0428/Graph-Cities/tree/main<br>

## **Languages, Tools and Libraries:**
**Python, Apache Spark, Django, HTML, CSS, JS, SQLite, Neo4j, 3D Force Graph**

## **Project Preview**

• Computed **PageRank on the entire Wikipedia dataset** until 01 Dec 2019, and built an **efficient search engine** that produces the top 10 pages for a given keyword, based on pagerank<br>
• The well-known PageRank algorithm, which is the **Power Iteration of the Matrix Formulation, brings external memory complications** as the number of webpages increases across the World Wide Web<br>
• The first major outcome of this project is **performing Distributed Processing** on a massive dataset such as the Wikidata dataset that has over 51 million vertices and 141 million edges, by leveraging Apache Spark<br>
• The second major outcome is the **application of Block Stripe Analysis which combats the memory limitations** posed by the Power Iteration method. It also alleviates and does not specifically need to handle deadends and spider traps.<br>
• Achieved **100% accuracy** for PageRank values and optimized the computation across 1000 partitions, resulting in a more efficient processing pipeline <br>
• The application successfully implements a search engine for all Wikipedia articles, with search results ranked by PageRank.<br>
• Summary of each article implemented using BART algorithm.<br>

<img width="1470" alt="pagesagexpert" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/e27ebd00-70b6-4e05-abef-8cb09c3190ca">
<img width="1015" alt="Screenshot 2024-01-10 at 7 02 01 PM" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/58d37e16-cc93-41e0-a763-2c64b982c086">

## Data Visualization
• Visualized Wikidata dataset (51 million vertices and 141 million edges) as a **Graph City**

<img width="766" alt="Screenshot 2023-12-03 at 5 49 00 PM" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/966f2310-7948-4d55-8356-c2f2c4b2b615">

• The application produces a **subgraph of the relevant search results**, showing how the resulting articles are connected (3D Force Directed Layout)<br>
<img width="310" alt="Screenshot 2024-01-10 at 7 00 14 PM" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/9ba29131-f38a-4811-b7c2-76b7dc43e64d"><br>
• Performed Text summarization on each article using BART technique<br>
• For each user, the application suggests articles for the user based on previous search history<be>
<br>
## **Steps to Run**
1. Download the project directory
2. On the command line, navigate to the project directory -> mds_app
3. Then enter `python manage.py runserver` (make sure python is already installed in your computer)
4. You will see an address like http://127.0.0.1:8000/ or with any other port number than 8000
5. Copy and paste that into your browser, the application should now run in your browser

**CONTRIBUTORS**
1. Rhema Marneni @RutgersUniversity-NewBrunswick
2. Bhargavi Chinthapatla @RutgersUniversity-NewBrunswick
3. Saman Ebrahimi @RutgersUniversity-NewBrunswick
