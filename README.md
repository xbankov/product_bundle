# product_bundle

Implementation of new recommendation engine component for the Conrad online shop.
The compontent recommends our customers products that might be interesting to buy together (bundles).

# Tasks:

1. Implement a solution of your choice for recommending product bundles e.g.
   rule-based, statistics, or ML-based solution. Please describe any reasoning behind
   your solution.
2. Provide a splitting to train and test datasets. Discuss possible different splitting
   criteria. What other splitting criteria would you choose if you could gather more
   features/data?
3. Discuss the size of the output list and how it can be decided per product
4. Discuss/implement any price computation per bundle e.g. the sum of products’
   prices
5. How would you evaluate the business impact of the solution and share the outcome
   with the internal stakeholders?
   Optional tasks:
6. Implement a regression model for the products’ prices (UnitPrice) prediction. Is the
   provided data sufficient to predict the price? What other data would you like to gather
   to improve your solution?
7. Your bundle's code is a great success and the Frontend team wants to use it in
   production. Implement a simple Rest API to serve the bundles with an endpoint
   getting as a parameter a product ID and returning a list of products and the price for
   the whole bundle. Ideally, provide a Dockerized version of the implemented API.

### Product Bundle

I have decided to first try rule-based approach using market basket analysis. The association rules were generated using apriori algorithm with the properly chosen hyperparameters, this could be improved using grid-search or other hyperparameter optimization. I have predicted bundles using these association rules.

### Splitting the dataset

I am used to split data in a 60/20/20 percent into train, validation and test set. Other ratios like 80/10/10 should be considered, giving more data for the training/rule-extraction and less for evaluation. Train and validation datasets are used to find the best hyperparameters for the models/rule-based solutions, while test set gives the prediction of how well will model behave in real world situations. Test should be used only to evaluate performance, not to debug the models/rules, that's why I used 3 splits. Evaluation certainty about the performance can be further improved by doing K-fold cross-validation.

I have decided to split data by InvoiceNo for rule-based system evaluation. As we don't use seasonality, product or customer data.

I also experimented with collaborative filtering and matrix factorization, in which case, I splitted data by customerID, as I wanted to use the customer informations to see if products can be clustered/bundled by the people who buy them.

If I wanted to incorporate seasonality, I would need dataset covering more than 1 year, to be able to model shopping behaviour during holidays, etc.

### Discuss the size of the output list and how it can be decided per product

In general I would put the boundary on how many items should be shown. There can be a machine learning model predicting how many additional items would specific customer buy additionally. For example average number of items purchased per Invoice of the customer could be a good guess. Without any other informations, using median number of items bought per invoice would be beneficial.

Also items could be added sequentially, one by one, and we can train model to see how probable it is that this bundle will be bought. We would add items until we are below threshold of "would be probably bough".

### Discuss/implement any price computation per bundle e.g. the sum of products’ prices

If I don't have access to current catalogue, I can always take the last purchase of the item. I need to take into account discounts. It could be beneficial to put slight discount on the bundle, given that customer would buy more, if he choses bundle, we should be still profiting.

### How would you evaluate the business impact of the solution and share the outcome with the internal stakeholders?

- The rule-based solution even though covering small space of potential bundles is successfull 40% of times in our validation dataset. Every 2nd and a half bundle will be bought. We will ease the customers experience by using only one click. Retention rate would be higher, and that we can measure using A/B test.

### Implement a regression model for the products’ prices (UnitPrice) prediction.

### Dockerized RestAPI
