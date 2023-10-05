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

I have provided three solutions:

1. Rule-based Apriori:

I chose rule-based apriori algorithm (market basket analysis) as a first choice. Even though association rules generated were covering very small number of possible bundles, they had very good precision during evaluation on training_df even though I couldn't find parameters that would find some bundles in valid_df and test_df. This should be undergo more detailed analysis to determine if the problem is small number of samples, misuse of the library or some other bug. This approach, however, disregards Description and UnitPrice of the products as well as any CustomerID information.

2. Collaborative Filtering:

I decided to implement item-based collaborative filtering in the simplest way possible, using item-user matrix one-hot encoding. Each row is a product vector made out of 0s or 1s for each customer. Cosine similarity (actually cosine distance) should give reasonable prediction of items bought together. This make use of different behavioural patterns of customers purchases.

3. Content-based Filtering:

At last I decided to also make use of similarity between products using textual Description and numerical UnitPrice. In a simple notebook I found that generated bundles/products are indeed very similar in a semantic sense.

#### Future Work:

- If CustomerID would be passed during inference as well, bundling could be then personalized.
- I also had an idea to create neural network with triplet loss, training using negative and positive pairs of bundles. However I focused on demonstrating basics of recommendation systems.
- For the same reason I did not choose other not so established SOTA research implementations, which should be also considered.

### Splitting the dataset

I splitted the data in a 60/20/20 ratio into train, validation and test set. Other ratios like 80/10/10 should be considered, giving more data for the training and less for evaluation. Train and validation datasets are used to find the best hyperparameters for the models solutions, while test set gives the prediction of how well will model behave in real world situations. Test should be used only to evaluate performance, not to debug the models/rules, that's why I used 3 splits. Evaluation certainty about the performance can be further improved by doing K-fold cross-validation.

I have decided to split data by InvoiceNo for all evaluations. As we don't use seasonality or customer data.

Splitting by CustomerID would be beneficial, if I had customerID during the inference process. Same with seasonality and additionaly I would assume longer recorded period would be necessary to really address changes during the 1 year period.

### Discuss the size of the output list and how it can be decided per product

- I believe this can be considered hyper-parameter of a model and should be subjected to the hyperparameter search, and optimal global size can be decided with the proper evaluation.
- Other option, would be to sequently look at the confidence of adding product into the bundle and stop at specific threshold.
- There should also be some upper limit, which we can get from the purchase statistics.
- I decided for rule-based system to choose biggest bundle, as those rule should be rare but precise. And for other I chose a bundle_size computed from the dataset and hyperparameter search.

### Discuss/implement any price computation per bundle e.g. the sum of products’ prices

If I don't have access to current price catalogue, I can take the last purchase of the item as a baseline. It could be beneficial to put slight discount on the bundle, given that customer would buy more products from us using a bundle. However, to determine that I believe I would need actual cost/profit data, to not undersell the items.

### How would you evaluate the business impact of the solution and share the outcome with the internal stakeholders?

- The rule-based solution even though covering small space of potential bundles is successfull 40% of times in our validation dataset. Every 2nd and a half bundle will be bought. We will ease the customers experience by using only one click. Retention rate would be higher, and that we can measure using A/B test.
- Additionaly collaborative filtering can give inside about other customers purchased products in a bundle, for example products for a same use (guitar pick + guitar capo + guitar stand), that may not share the same textual description or price but are often bought together.
- And finally content-based filtering shows similar products together, therefore it servers users searching for the same type of the product in a collection, for example different colors, candle scents, etc.

### Implement a regression model for the products’ prices (UnitPrice) prediction.

I implemented simple neural network using encoded description using distillbert and encoded datetime fastai feature engineering to predict the price. I believe test MSE Loss of 0.4461 shows that I am in the right direction, and I decided not to improve it anymore for the ske of time. I also decide not to add it to the REST_API, as I believe that was not wanted from the FrontEnd team, but can be easily adjusted if needed.

### Dockerized RestAPI RUN

- Clone git repo:

`git clone git@github.com:xbankov/product_bundle.git`

- Change directory

`cd product_bundle`

- Build a docker image:

`docker build -t bundle_api .`

- Run a docker image:

`docker run -p 8888:8888 bundle_api`

- Go to an API an look for a bundle for product 22423

`http://0.0.0.0:8888/bundles/22423`

### Run

To generate all files in data and in static run `prepare.sh`

### Disclaimer

- Current evaluation shows that all the technique are either wrong, or there is a bug in my evaluation, I strongly suspect the second.
- It would be possible to formulate the task differently, where the model do not generate the whole bundle, just suggest the single item to add into a bundle. Current methods could be easily adjusted for that purpose and the evaluation as well.

### Source code structure

- Source code
- bundling\_\*.py - functions used during evaluation and generation
- evaluate\_\*.py - scripts to find hyperparameters, save the best params, evaluate metrics and save the bundles from the best
- pricing\_\*.py - Regression model and LastPurchase "model"

### Notebooks

- I have experimented with the code using notebooks, these are in general not currated or runnable.
- The most interesting was gnn_experiment.ipynb, where I tried to use Graph Neural Network, unfortunately, my PC was not able to handle the calculations, therefore I stopped trying to make it work.
- In `analyse_bundles.ipynb` there is an random example of created bundles
