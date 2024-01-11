# PageSageXpert

(This project was done as a part of my coursework for "CS543: Massive Data Storage, Retrieval and Deep Learning" in Fall 2023 under Prof. James Abello at Rutgers University)
https://docs.google.com/document/d/15nH_5_L9TQo82f4hSuKDSVl_tmNtUKao/edit
A Wikipedia Application based on PageRank

<img width="781" alt="pagesagexpert_outline" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/18c78201-22e8-4e5f-b4b1-eef0958174f3">

_Dataset_: https://www.kaggle.com/datasets/kenshoresearch/kensho-derived-wikimedia-data<br>
_Youtube link to Graph City_: https://www.youtube.com/watch?v=azRk8F7-KRE<br>
_Original Graph City Repository_: https://github.com/endlesstory0428/Graph-Cities/tree/main<br>

**Python, Apache Spark, Django, HTML, CSS, JS, SQLite, Neo4j, 3D Force Graph**

• Computed PageRank on entire wikipedia dataset until 01 Dec 2019, and built an efficient search engine that produces top 10 pages for a given keyword, based on pagerank

<img width="1470" alt="pagesagexpert" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/e27ebd00-70b6-4e05-abef-8cb09c3190ca">
<img width="1015" alt="Screenshot 2024-01-10 at 7 02 01 PM" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/58d37e16-cc93-41e0-a763-2c64b982c086">

• Visualized wikidata dataset (51 million vertices and 141 million edges) as a Graph City

<img width="766" alt="Screenshot 2023-12-03 at 5 49 00 PM" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/966f2310-7948-4d55-8356-c2f2c4b2b615">

• Achieved 100% accuracy by implementing the Block Stripe Analysis technique for PageRank to optimize the computation across 1000 partitions, resulting in a more efficient processing pipeline <br>
• The application produces a subgraph of the relevant search results, showing how the resulting articles are connected (3D Force Directed Layout)<br>
<img width="310" alt="Screenshot 2024-01-10 at 7 00 14 PM" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/9ba29131-f38a-4811-b7c2-76b7dc43e64d"><br>
• Performed Text summarization on each article using BART technique<br>
• For each user, the application suggests articles for the user based on previous search history<br>


**CONTRIBUTORS**
1. Rhema Marneni @RutgersUniversity-NewBrunswick
2. Bhargavi Chinthapatla @RutgersUniversity-NewBrunswick
3. Saman Ebrahimi @RutgersUniversity-NewBrunswick
