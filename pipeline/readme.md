# Construction Pipeline

💡 For research purposes, we open-sourced part of the code for constructing the dataset. However, the current structure of the code is not very tidy. We will organize it in the future.

💡 Just For Reference

# Step

## Preprocessing
1. collect_news_api.py  --get news api
2. news_crawler.py --get news body
3. filter_news_token.py -- filter: tokens > 1048
4. filter_news_dupicate.py -- filter: Remove duplicates(body) | datasource/news_filter_dup.json
5. fact_or_opinion.py -- generate: Filter out factual sentences |  datasource/news_filter_fact.json
6. filter_same_fact.py --  filter: Keep the unique claim   | datasource/news_filter_fact_out_same.json
   
## Grouping News For Multihop Question
### Group By Keywords
7. generate_key.py -- get keywords  | datasource/news_filter_key.json
8. group_by_key.py -- group by key 

### Group By Entity
7. generate_entity.py -- get entity  |

### Generate Claims 
7. prepare_claims.py
   
You can also use claim_topic/claim_target to group the news

## Generate Q&A