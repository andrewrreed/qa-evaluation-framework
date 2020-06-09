# Question Answering Retriever Evaluation



## General Approach

1. Setup test harness
2. Establish a baseline - use basic ES BM25 on FULL Wikipedia Articles





## Tech Debt

#### Important Upfront

- [ ] 

- [ ] Remove questions with >5 words in short answer?

- [x] Add document_title to ES index

- [ ] Do we need to dedup articles? If so, how to properly? (Dedup on title taking max id?)

  - ```
    https://en.wikipedia.org//w/index.php?title=2017_NFL_season&amp;oldid=816087307
    https://en.wikipedia.org//w/index.php?title=2017_NFL_season&amp;oldid=816254311
    ^^ These both have the exact same date of update: This page was last edited on 19 May 2020, at 00:40 (UTC) ????????????
    ```

  ![image-20200609094714358](/Users/areed/Library/Application Support/typora-user-images/image-20200609094714358.png)

- [ ] 40971 of the 87407 are unique...
  
- [ ] How to handle ES reserved characters in input string? (Ex / is reserved)
  
  - https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-query-string-query.html

#### TO DO's

- [ ] Refactor into modules
- [ ] Refactor data prep into one single script (download data, load, wrangle, save to local)



