# PageSageXpert
A Wikipedia Application based on PageRank

<img width="781" alt="pagesagexpert_outline" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/18c78201-22e8-4e5f-b4b1-eef0958174f3">

Dataset Link: https://www.kaggle.com/datasets/kenshoresearch/kensho-derived-wikimedia-data

**Python, Apache Spark, Django, HTML, CSS, JS, SQLite, Neo4j, 3D Force Graph**

• Computed PageRank on 5.3 million wikipedia pages and built an efficient search engine that produces top 10 pages for a given keyword, based on pagerank

<img width="1470" alt="pagesagexpert" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/e27ebd00-70b6-4e05-abef-8cb09c3190ca">

• Visualized wikidata dataset (51 million vertices and 141 million edges) as a Graph City

<img width="766" alt="Screenshot 2023-12-03 at 5 49 00 PM" src="https://github.com/rhemaMarneni/PageSageXpert/assets/67055118/966f2310-7948-4d55-8356-c2f2c4b2b615">

• Achieved 100% accuracy by implementing the Block Stripe Analysis technique for PageRank to optimize the
computation across 1000 partitions, resulting in a more efficient processing pipeline
• The application produces a subgraph of the relevant search results, showing how the resulting articles are connected (3D Force Directed Layout)
• Performed Text summarization on each article using BART technique
• For each user, the application suggests articles for the user based on previous search history
