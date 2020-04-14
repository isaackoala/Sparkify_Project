# 毕业项目

### 数据挖掘工程师直通班

##### 邵苏	2020-4-14



# 定义

### 项目概述

**顾客流失**是会影响到所有企业的痛苦现实。即使是最大或最成功的公司也无法避免客户的背叛。为了保持业务的可持续增长，我们需要了解哪些因素正导致着客户忠诚度的下降。

本项目针对著名的线上音乐网站Sparkify发生的顾客流失问题，对其所提供的12GB有关用户的网上操作信息进行数据分析：分析与顾客流失相关的用户信息特征，并建模预测顾客流失，最终由此进一步确定可影响顾客流失的重要信息特征，并提供相应的对策。由于数据量巨大，本项目将对其中128MB的数据进行先行分析，建模，评价，以便为成功的进一步分析全数据集，选择模型提供有力的指导。本项目采用Spark处理数据集，已达到快速处理的效果。数据集输入信息涉及：用户信息、歌曲信息、网页状态信息、注册及退出信息等，具体讲于数据分析中进一步阐述。

### 问题描述

本项目的目标是要通过对现有128MB数据集的分析评估以确定顾客流失率及确定对于顾客流失起主要影响的顾客信息特征，以便为Sparkify 制定应对策略提供指导和参考。项目具体涉及几方面任务：

1. 下载Spariky数据集。
2. 探索性分析数据集，并确定与顾客流失相关的特征。
3. 对相关特征进行聚合并重组。
4. 设计模型进行分类及进一步调优。
5. 分析分类指标及确定主要影响特征。

本项目希望能够找到较优的模型用以进一步对全数据集分析训练及测试，以获得相对较好的分类指标。实现项目目标。

### 评价指标

本项目主要使用两类评价指标对模型进行评价，分别为准确率和F1值。

1. **准确率**是一种常见的二分类指标。其定义如下：	

    Accuracy = (True Positives + True Negatives)/dataset size

​		准确率指标可直接反映出顾客流失或不流失的状态，是非		常有效的评价指标。但对于正负样本不均衡的情况，会出		现评价不准确。

 2. **F1值**被用来弥补解决准确率存在的欠缺问题。其定义如下：
 
    P = True Positives/True Positves *
    False Positives 
    

   
    R = True Positives/True Positives*False Negatives
    
    

    
    F1 = 2PR/(P+R)
    

    F1值对精确率P和召回率R进行了加权调和平均，既考虑到查准率又考虑到了查全率，以此来评价模型实际的效果可降低正负样本不均衡所带来的影响，得到理想的效果。

本项目是典型的二分类问题，因此使用准确率来评价模型分类的优劣，同时为了保证较高的查准率和查全率，F1值也被用来对模型进行评价，以指导模型选择。



# 分析

### 探索性数据分析

本项目中，目标数据集提供了用户、歌曲、页面、注册状态等多项信息具体如下：

```
root
 |-- artist: string (nullable = true)
 |-- auth: string (nullable = true)
 |-- firstName: string (nullable = true)
 |-- gender: string (nullable = true)
 |-- itemInSession: long (nullable = true)
 |-- lastName: string (nullable = true)
 |-- length: double (nullable = true)
 |-- level: string (nullable = true)
 |-- location: string (nullable = true)
 |-- method: string (nullable = true)
 |-- page: string (nullable = true)
 |-- registration: long (nullable = true)
 |-- sessionId: long (nullable = true)
 |-- song: string (nullable = true)
 |-- status: long (nullable = true)
 |-- ts: long (nullable = true)
 |-- userAgent: string (nullable = true)
 |-- userId: string (nullable = true)
```

在这众多特征中涉及用户的交互信息，也有注册及注销的时间信息，这些信息都对预测顾客流失起到了重要的作用，共有286500条记录。

其中特征Page对于理解用户行为起到了至关重要的作用。其中包含如下值：

*Logout, Save Settings, Roll Advert, Settings, Submit Upgrade, Cancellation Confirmation, Add to Playlist, Home, Upgrade, Submit Downgrade, Help, Add Friend, Downgrade, Cancel, About, Thumbs Down, Thumbs Up, Error*

**顾客流失量：**在这众多的值中，Cancellation Confirmation和downgrade是最能反映出用户对于Spariky所提供服务的负面态度的，也最为直接的预示顾客流失的信息。因此在Cancellation Confirmation 事件被用来定义为客户流失。因为该事件在付或免费客户身上都有发生。通过对该事件的标注，进一步对发生该事件的用户进行标注，以此来确定顾客流失与否。也将成为模型唯一输出。这一特征对于判断顾客是否流失至关重要。本项目中将发生顾客流失定义为1，未发生的情况定义为0。

此外对于网页信息来说最为重要的两个特征为userId和seesionId，一个是有关用户的唯一信息，可作为主键，而另外一个则是网页运行所特有的信息，与用户在网页上停留时间，所进行的操作都有关系。经探索发现userId和seesionId中存在着重复值，也存在着空值，总共为8346个。虽然相对数据总量不算多，但依然会影响到模型的分析。

为便于阅读，registration和ts中的时间戳信息也可转化为datetime数值。

### 数据可视化

在明确定义了顾客流失的特征指标后，为了直观地观察顾客忠诚度表现，项目对如下四方面的数据进行了可视化分析：

1. 顾客所听歌曲数目分析
![image](https://github.com/isaackoala/Sparkify_Project/blob/master/Life.png style = "zoom:150%")
   
   如图所示，选择继续使用Spariky的用户和选择不再使用的用户在所听的歌曲数目上差异并不大。但很明显女性的忠诚度更为高一些，而同时女性的用户的听歌数量分布也更为广些，有一二百首的，也有超过8000首的。

2. 顾客选择注销可能性分析

   <img src="\C:\Users\Isaac\Pictures/Likeness.png" alt="Alt text" style="zoom:150%;" />

   如图所示，Sparkify成功地赢得了大多数用户的青睐和喜爱，并选择继续接受服务。在选择注销的用户中男性数量比女性数量多了近三成以上。这说明Sparkify在运营中依然无法完成赢得一小部分用户的喜爱，尤其是男性用户。至于这部分用户的是何原因选择离开，由于缺乏相关数据，目前很难确定。需要进一步采集用户反馈获得。

3. 付费与免费用户分析

   <img src="\C:\Users\Isaac\Pictures/Paid.png" alt="Alt text" style="zoom:150%;" />

   如图所示，在付费与免费两大用户中，有更多的付费用户选择注销停止接受Sparkify的音乐服务。这里就存在一个值得探讨的问题，为什么会有更多的付费用户选择停止接受服务呢？虽然目前没有更多用户信息以兹证明，但可以推测的是，付费用户普遍对Sparkify所提供的服务有着更高的要求。因此值得思考的是Sparkify可以提供哪些对于付费用户来说极有价值的服务，使他们对此认可并欣然愿意继续支付比免费用户更多的费用来享受会员服务呢？可考虑进一步反馈信息的探索，也可对付费用户的音乐喜欢进行分析研究。

4. 用户生命周期分析

   <img src="\C:\Users\Isaac\Pictures/Life.png" alt="Alt text" style="zoom:150%;" />

   如图所示，忠诚度高的用户比选择停止接受服务的用户普遍有着更高的用户生命周期（从用户注册到注销账号之间的时间）。而无论哪类用户，其男性女性用户生命周期差异并不是很大。忠诚度高的用户的生命周期大多在50-150天之间，而选择注销的用户的生命周期择普遍低于仅为一半。 为何用户的生命周期仅为150天左右，如何提高满足用户不断变化的音乐享受要求是Sparkify需要进一步考虑的问题。

由如上数据可视化分析可知，性别，歌曲收听数量，用户生命周期等特征与用户流失用着密切关系，可作为进一步建模分析的特征，其他相关特征也可进一步考虑，于下一部分方法步骤中具体阐述。



# 方法步骤

### 数据预处理

本项目数据预处理涉及数据清洗，特征工程和数据转换。

1. 有关重复数据和缺失数据的处理已在数据探索部分进行了，可见详见以上分析部分，所以在此不再重复。

2. 特征工程部分主要是对数据集在与分类有关的特征聚合与合并。其中涉及的特征有：
   - life_time：用户注册时间和注销时间的差值，表示使用的使用Sparkify所提供服务的时间。考虑到用户可能多次注册，本项目仅选择最大差值最为用户的生命周期。本特征能反映用户使用Sparkify服务的长短，可直接反映出其客户忠诚度，评判顾客流失与否。

   - num_songs: 用户听歌的总数，表示在册期间同一userId账户的听歌数量。本特征能反映用户使用Sparkify作为平台听歌的数量，可直接反映出其对Sparkify的使用频率。基于当用户用Sparify听歌越多就对该平台的使用越适应，并且更换其他的平台可能性就越低的假设，此特征也可反映出其忠诚度，作为评判顾客流失的指标。

   - num_thumb_up：用户点赞数，表示在册期间同一个userId账户的点赞总数。

   - num_thumb_down：用户点差数，表示在册期间同一个userId账户的点差总数。

     以上两个特征可反映出用户对歌曲的感受，喜欢或是不喜欢。听歌永远是相对被动的行为，但是表达感受则是更为直接的行为。这直接为用户的感受有关，但如果用户愿意在Sparkify平台以点赞或点差的方式表达，既能反映其情感，同时又能反映出其对Sparkify平台的使用的熟悉程度和相关服务的参与程度。用户越适应于用Sparkify所提供的服务来表达自己对歌曲的见解，就越不会轻易更改听歌平台。因此可作为评判顾客流失的指标。当然如果当某用户的点差数远远多于点赞数时，可从一方面反映出其对Sparkify平台提供歌曲或服务的不满，可判断为其停止服务意愿的增加。

   - add_to_playlist：添加歌单数，表示在册期间同一个userId账户的将某歌曲添加至歌单的次数。本特征与用户适应于使用Sparkify平台提供的功能有关，并且为了便于再次听歌而将歌曲直接加入歌单生成自己的歌单。那么通常这种情况反映出用户不但适应于Sparkfiy所提供的服务，并开始着手使这些服务个性化。用户越是愿意在这些方面投入时间和精力，其就越不容易轻易更改听歌平台。因此可用于判断顾客流失与否。

   - add_friend：添加朋友数，表示在册期间同一个userId账户的发生添加朋友的行为的次数。本特征反映出用户正开始适应同Sparkify平台发展自己的人脉，与自己有相同音乐偏好的人，可以说正在进一步使Sparkify建设成为用户社区。当用户有越来越多的来自Sparkify，有着同样喜好的朋友，更改音乐平台所带来的损失可能不单只是时间金钱的成本，还有大量的人脉和情感成本。因此在Sparkify上添加朋友越多，用户就越不太可能流失。

   - total_listen：听歌总时长，表示在册期间同一个userId账户的听歌时长的总量。本特征同样从一个方面反映出用户适应Sparkify的频率。除以life_time则可直接获得用户每天使用Sparkify听歌时长。这个指标越高则反映用户使用频率越高，同时也反映出用户对于平台的依赖程度越高。可以推得，其更改音乐平台的可能性也就越低。

   - avg_songs_played：平均听歌数，表示用户在册期间同一个userId账户平均连续听歌数目。本特征在于反映用户使用Sparkify听歌的程度。Sparkify除了提供音乐服务外，同时还提供其他的服务，比如更改设置，帮助，注册，升降级等。然而目前Sparkify主要的服务依然为音乐服务，官方也以用户使用音乐服务的程度来评定平台效率。因此若用户评价连续听歌数目越多，那么其就越纯粹地接受Sparkify所提供的服务，再以其在其他特征的反映就更为正确地确定用户的忠诚度。而另一方面，练习评价听歌数目本身就足以反映用户对音乐平台的使用频率和依赖程度。因此可作为评判顾客流失与否的指标。

   - gender：性别，表示用户在册期间同一userId账号的性别。本特征是个有关人口统计的特征，用以反映性别在顾客忠诚度上的差异。可为男女用户的差异化服务提供有效的信息和依据。

   - num_artist：艺术家数目，表示用户在册期间同一个userId账户的连续听歌时歌曲的艺术家数目。本特征可有效反映用户偏好，即用户在连续听歌时，艺术家的分布情况。对于avg_songs_played可以得到用户是否较为集中听某几个艺术家的歌，或是听歌比较随意者对于了解用户音乐偏好很有帮助。可由此提供适应用户的推荐，以增加其用户忠诚度。

   - label：顾客流失量，详见分析部分的定义。

   分别对userId以```groupby```和```agg```分类统计以上特征，同时使用```dropDupliates```对重复记录进行清理。此外对于分类数据gender，直接以```replace```更换为男性为0，女性为1。label的分类数据已经在分析步骤中更换为数值型，这里不再重复。最终以``join``，``outer``为参数合并所有特征，以``fillna``填充所有缺失值，获得225*11的dataFrame数据集作为实验用数据集。

3. 为了便于模型训练分类，此处还将特征工程所获dataFrame数据集进行数据转换。其中包括数据向量化和数据标准化。

   - 数据向量化，将所有输入特征，即除label之外的所有特征通过``vecAssembler``转化为向量矩阵。

     ```python
     # vector assembler
     cols = ['life_time', 'num_songs', 'num_thumb_up', 'num_thumb_down', 		'add_to_playlist','add_friend', 'total_listen','avg_songs_played', 		'gender', 'num_artist']
     
     # merges multiple columns into a vector column.
     vecAssembler = VectorAssembler(inputCols = cols, outputCol = 'NumFeatures')
     # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler
     
     data = vecAssembler.transform(data)
     # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=transform#pyspark.ml.feature.VectorAssembler.transform
     
     ```

   - 数据标准化，为了减少因不同特征向量范围差异过大带来的训练和测试偏差，用``standarScaler``对以转化的向量矩阵以标准差进行标准化。

     ```python
     # Standardizes features by removing the mean and scaling to unit variance using column summary statistics on the samples in the training set
     standardScaler = StandardScaler(inputCol = 'NumFeatures', outputCol = 'features', withStd = True)
     
     model = standardScaler.fit(data)
     
     data = model.transform(data)
     # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=standardscaler#pyspark.ml.feature.StandardScaler
     ```

完成上述数据预处理所得的数据集以10个特征的向量矩阵为输入和label为输出，可进行下一步模型训练与测试。

### 代码实现

在代码实现步骤中，主要涉及数据分割，天真预测和模型设计。

1. 数据以``randomSplit``分割为60%的训练集，20%的验证集和20%的测试集。

   ```python
   # divid data into 6:2:2 
   train, validation, test = data.randomSplit([0.6, 0.2, 0.2], seed = 42)
   # http://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=randomsplit#pyspark.sql.DataFrame.randomSplit
   ```

2. 为了确定能够对模型设计与选择提供有力的指导，本项目在设计模型之前选择使用天真预测确定基准预测值。本项目为二分类问题，因此设置两个天真预测，分别预测churn = 1（顾客流失）和churn = 0（顾客未流失）的情况，以准确率和F1值检测。代码如下：

   ```python
   # baseline model for churn = 1
   result_baseline_1 = test.withColumn('prediction', lit(1.0))
   # http://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=withcolumn#pyspark.sql.DataFrame.withColumn
   # http://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=lit#pyspark.sql.functions.lit
   
   evaluator = MulticlassClassificationEvaluator(predictionCol = 'prediction')
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator
   
   print("Test Set 1 Metrics: ")
   print('Accuracy: {}'.format(evaluator.evaluate(result_baseline_1, {evaluator.metricName: 'accuracy'})))
   print('F1: {}'.format(evaluator.evaluate(result_baseline_1, {evaluator.metricName: 'f1'})))
   
   # baseline model for churn = 0 
   result_baseline_0 = test.withColumn('prediction', lit(0.0))
   # http://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=withcolumn#pyspark.sql.DataFrame.withColumn
   # http://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=lit#pyspark.sql.functions.lit
   evaluator = MulticlassClassificationEvaluator(predictionCol = 'prediction')
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator
   
   print("Test Set 0 Metrics: ")
   print('Accuracy: {}'.format(evaluator.evaluate(result_baseline_0, {evaluator.metricName: 'accuracy'})))
   print('F1: {}'.format(evaluator.evaluate(result_baseline_0, {evaluator.metricName: 'f1'})))
   ```

3. 为了更好地为本项目的分类问题设计模型，分别选择Logistic Regression（逻辑回归），Gradient-Boosted Trees（梯度提升树），Support Vector Machine（支持向量机）和Random Forest（随机森林）作为模型训练并验证现有数据集，使用准确率来和F1评定，同时还考虑时间消耗成本。

   由于最终数据集的数据量只有225条, 相对来说讲很少对于大多数的分类器来说显得很少，不足以很好的提供训练数和验证数。很容易出现过拟合或欠拟合的情况，这将极大的影响测试结果。为此选用3-fold交叉验证的方式增加数据量，以便获得比较好的测试结果，用于评定和改进模型。

   - Logistic Regression以``LogisticRegression``作为分类器，以F1值作为评测指标对进行训练集进行训练，以获得训练耗时。同时还使用``ParaGridBuilder``设置参数网格，使用``CrossValidator``进行3-fold交叉验证。以训练所得模型进行验证，以准确率和F1值评定。代码如下：

   ```python
   # Initialize Classifier with 10 iterations
   lr = LogisticRegression(maxIter = 10)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=logisticregression#pyspark.ml.classification.LogisticRegression
   
   # Set f1 score as evaluator
   evaluator = MulticlassClassificationEvaluator(metricName = 'f1')
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator
   
   # builder for a param grid used in grid search-based model selection.
   grid  = ParamGridBuilder().build()
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=paramgrid#pyspark.ml.tuning.ParamGridBuilder
   
   # 3-fold cross validation performs LogisticRegression 
   cv_lr = CrossValidator(estimator = lr, estimatorParamMaps = grid,\
                          evaluator = evaluator, numFolds = 3)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator
   
   # start time
   start = time()
   # fit trainset
   model_lr = cv_lr.fit(train)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=fit#pyspark.ml.classification.LogisticRegression.fit
   # end time
   end = time()
   
   # print the time span of the process
   print('The training process took {} seconds'.format(end - start))
   
   # transform validationset
   result_lr = model_lr.transform(validation)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=transform#pyspark.ml.classification.LogisticRegressionModel.transform
   
   # set evaluator for prediction
   evaluator = MulticlassClassificationEvaluator(predictionCol ='prediction')
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator
   
   # Print Accuacy and F1 score of Logistic Regression
   print('Logistic Regression Metrics: ')
   print('Accuracy: {}'.format(evaluator.evaluate(result_lr, {evaluator.metricName: 'accuracy'})))
   print('F1: {}'.format(evaluator.evaluate(result_lr, {evaluator.metricName: 'f1'})))
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=evaluate#pyspark.ml.evaluation.MulticlassClassificationEvaluator.evaluate
   ```

   - Gradient-Boosted Tree以``GBTClassifier``作为分类器，以F1值作为评测指标对进行训练集进行训练，以获得训练耗时。同时还使用``ParaGridBuilder``设置参数网格，使用``CrossValidator``进行3-fold交叉验证。以训练所得模型进行验证，以准确率和F1值评定。代码如下：

   ```python
   # Initialize Classifier with 10 iterations and 42 samples
   gbt = GBTClassifier(maxIter = 10, seed = 42)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=gbtclassifier#pyspark.ml.classification.GBTClassifier
   
   # Set Evaluator
   evaluator = MulticlassClassificationEvaluator(metricName = 'f1')
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator
   
   # build paramGrid
   grid = ParamGridBuilder().build()
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=paramgrid#pyspark.ml.tuning.ParamGridBuilder
   
   # 3-fold cross valiation with paramgrid
   cv_gbt = CrossValidator(estimator = gbt,evaluator = evaluator,\
                           estimatorParamMaps = grid, numFolds = 3)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator
   
   # Set start time
   start = time()
   # fit trainset
   model_gbt = cv_gbt.fit(train)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=fit#pyspark.ml.classification.LogisticRegression.fit
   # end time
   end = time()
   
   # print the time span of the process
   print('The training process took {} seconds'.format(end - start))
   
   # transform validation set
   result_gbt = model_gbt.transform(validation)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=transform#pyspark.ml.classification.LogisticRegressionModel.transform
   
   # set evaluator for prediction
   evaluator = MulticlassClassificationEvaluator(predictionCol ='prediction')
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator
   
   # print accurcay & f1 score of GBT
   print('Gradient Boosted Trees Metrics: ')
   print('Accuracy: {}'.format(evaluator.evaluate(result_gbt, {evaluator.metricName: 'accuracy'})))
   print('F1: {}'.format(evaluator.evaluate(result_gbt, {evaluator.metricName: 'f1'})))
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=evaluate#pyspark.ml.evaluation.MulticlassClassificationEvaluator.evaluate
   ```

   - Support Vector Machine以``LinearSVC``作为分类器，以F1值作为评测指标对进行训练集进行训练，以获得训练耗时。同时还使用``ParaGridBuilder``设置参数网格，使用``CrossValidator``进行3-fold交叉验证。以训练所得模型进行验证，以准确率和F1值评定。代码如下：

   ```python
   # Initialize Classifier with 10 iterations
   svm = LinearSVC(maxIter = 10)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=linearsvc#pyspark.ml.classification.LinearSVC
   
   # Set Evaluator
   evaluator = MulticlassClassificationEvaluator(metricName = 'f1')
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator
   
   # build paramGrid
   grid = ParamGridBuilder().build()
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=paramgrid#pyspark.ml.tuning.ParamGridBuilder
   
   # 3-fold cross valiation with paramgrid
   cv_svm = CrossValidator(estimator = svm, evaluator = evaluator,\
                           estimatorParamMaps = grid, numFolds = 3)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator
   
   # get start time
   start = time()
   # fit trainset
   model_svm = cv_svm.fit(train)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=fit#pyspark.ml.classification.LogisticRegression.fit
   # end time
   end = time()
   
   # Print the time span of the process 
   print('The training process took {} seconds'.format(end - start))
   
   # transform validation set
   result_svm = model_svm.transform(validation)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=transform#pyspark.ml.classification.LogisticRegressionModel.transform
   
   # set evaluator for prediction
   evaluator = MulticlassClassificationEvaluator(predictionCol ='prediction')
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator
   
   # print accuracy and f1 score of SVM
   print('Support Vector Machine Metrics: ')
   print('Accuracy: {}'.format(evaluator.evaluate(result_svm, {evaluator.metricName: 'accuracy'})))
   print('F1: {}'.format(evaluator.evaluate(result_svm, {evaluator.metricName: 'f1'})))
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=evaluate#pyspark.ml.evaluation.MulticlassClassificationEvaluator.evaluate
   ```

   - Random Forest以``RandomForestClassifer``作为分类器，以F1值作为评测指标对进行训练集进行训练，以获得训练耗时。同时还使用``ParaGridBuilder``设置参数网格，使用``CrossValidator``进行3-fold交叉验证。以训练所得模型进行验证，以准确率和F1值评定。代码如下：

   ```python
   # Initialize Classifier
   rf = RandomForestClassifier()
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=randomforest#pyspark.ml.classification.RandomForestClassifier
   
   # Set Evaluator
   evaluator = MulticlassClassificationEvaluator(metricName = 'f1')
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator
   
   # build paramGrid
   grid = ParamGridBuilder().build()
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=paramgrid#pyspark.ml.tuning.ParamGridBuilder
   
   # 3-fold cross validation
   cv_rf = CrossValidator(estimator = rf, evaluator = evaluator,\
                          estimatorParamMaps = grid, numFolds = 3)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator
   
   # set start time
   start = time()
   # fit train set
   model_rf = cv_rf.fit(train)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=fit#pyspark.ml.classification.LogisticRegression.fit
   # end time
   end = time()
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=avgmetric#pyspark.ml.tuning.CrossValidatorModel.avgMetrics
   
   # print the time span of the process
   print('The training process took {} seconds'.format(end - start))
   
   # transform validation set
   result_rf = model_rf.transform(validation)
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=transform#pyspark.ml.classification.LogisticRegressionModel.transform
   
   # set evaluator for prediction
   evaluator = MulticlassClassificationEvaluator(predictionCol ='prediction')
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator
   
   # print accuracy and f1 score of RF
   print('Random Forest Metrics: ')
   print('Accuracy: {}'.format(evaluator.evaluate(result_rf, {evaluator.metricName: 'accuracy'})))
   print('F1: {}'.format(evaluator.evaluate(result_rf, {evaluator.metricName: 'f1'})))
   # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=evaluate#pyspark.ml.evaluation.MulticlassClassificationEvaluator.evaluate
   ```

以上模型分别获得不同评价指标，对比数据，结果可用语下一步模型改进中。

### 模型改进

通过以上的四个模型的测试，获得四组准确率，F1值和时间损耗，结果如下：

- 逻辑回归模型的准确率为: 0.7959 & F1 score: 0.7871, using 524.71 seconds。
- 梯度提升树模型的准确率为: 0.7143 & F1 score:  0.7143, using 1088.89 seconds。
- 支持向量机模型的准确率为: 0.7959 & F1 score: 0.7055, using 741.00 seconds。
- 随机森林模型的准确率为: 0.7959 & F1 score: 0.7871, using 587.97 seconds。

通过比较上述数据，以目前小数据量的数据集上的表现上来看，逻辑回归和随机森林同时达到了最高的准确率和F1值，并且耗时几乎差不多。梯度提升树模型的准确率和F1值一致，且高于支持向量机的F1值，但时间开销是目前四个模型中最大的，几乎为随机森林的一倍。考虑到12G的全数据下，逻辑回归将显出更大的欠拟合，和支持向量机一样会大幅度增加时间开销。梯度提升树和随机森林被选择为进一步调优的模型，同时其算法所带的功能可直接反应出各特征对于分类的重要性，这将对进一步优化特征选择起到极大的作用。

本部分将同时对梯度提升树模型和随机森林模型进行调参。由于这两个模型的参数不尽相同，调参选择也有所区别。

1. 梯度提升树模型选用maxDepth和maxIter进行调优，取值范围分别为5-10和10-15。以这两维度构建参数网格。同样以3-fold交叉验证对训练集进行训练，以``avgMetrics``获得最优的评价指标评价数，以``getEstimatorParaMaps``获得对应参数。以最优参数代入模型重新训练并测试，以准确率和F1值评价。代码如下：

```python
# initalize GBT classifier 
gbt = GBTClassifier()

# build paramGrid in 5-10 maxDepth and 10-15 maxIter
paramGrid_gbt = ParamGridBuilder()\
    .addGrid(gbt.maxDepth, [5,10])\
    .addGrid(gbt.maxIter, [10,15])\
    .build()
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=paramgrid#pyspark.ml.tuning.ParamGridBuilder

# set evaluator
f1_evaluator = MulticlassClassificationEvaluator(metricName = 'f1')
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator

# 3-fold cross validation
crossval_gbt = CrossValidator(estimator = gbt, \
                             estimatorParamMaps = paramGrid_gbt,\
                             evaluator = f1_evaluator,\
                             numFolds = 3)
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator

# fit train set
cvModel_gbt = crossval_gbt.fit(train)

# get the best f1 score
cvModel_gbt.avgMetrics
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=avgmetric#pyspark.ml.tuning.CrossValidatorModel.avgMetrics
# the best f1 scores are higher than the prelimiary test

# get the corresponding maxDepth and maxIter of the best f1 score
cvModel_gbt.getEstimatorParamMaps()
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=getestimatorparammaps#pyspark.ml.tuning.CrossValidator.getEstimatorParamMaps

# best GBT model with maxDepth 5 and maxIter 10
gbt_best = GBTClassifier( maxDepth = 5, maxIter = 10, seed =42)
# fit train set
gbt_best_model = gbt_best.fit(train)
# transform test set 
results_final = gbt_best_model.transform(test)

# final resuls metrics from GBT
evaluator = MulticlassClassificationEvaluator(predictionCol = 'prediction')
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator

# Print accuracy and f1 score
print("Test Set Metrics: ")
print('Accuracy: {}'.format(evaluator.evaluate(results_final, {evaluator.metricName: 'accuracy'})))
print('F-1 Score: {}'.format(evaluator.evaluate(results_final, {evaluator.metricName: 'f1'})))
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=evaluate#pyspark.ml.evaluation.MulticlassClassificationEvaluator.evaluate
```

maxDepth = 5 和maxIter = 10为最优参数，因此代入模型，进行训练和最终测试。

同时以``featureImportances``获得最优模型中的各输入特征值的重要性，以反映其对判断顾客流失的影响程度。

2. 随机森林模型选用maxDepth和numTrees进行调优，取值范围分别为5-50和10-100。以这两维度构建参数网格。同样以3-fold交叉验证对训练集进行训练，以``avgMetrics``获得最优的评价指标评价数，以``extractParaMap``获得对应参数。以最优参数代入模型重新训练并测试，以准确率和F1值评价。代码如下：

```python
# Hyperparams Tuing Random Forest
rf = RandomForestClassifier()
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=randomforest#pyspark.ml.classification.RandomForestClassifier

# Set Evaluator
evaluator = MulticlassClassificationEvaluator(metricName = 'f1')
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator

# build paramGrid with 5-50 maxDepth and 10-100 numTrees
paramGrid_rf = ParamGridBuilder().addGrid(rf.maxDepth, [5,50])\
                .addGrid(rf.numTrees, [10,100]).build()
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=paramgrid#pyspark.ml.tuning.ParamGridBuilder

# 3-fold cross validation
cv_rf = CrossValidator(estimator = rf, evaluator = evaluator,\
                       estimatorParamMaps = grid, numFolds = 3)
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator

# fit train set
cvModel_rf = cv_rf.fit(train)
# get the best f1 
cvModel_rf.avgMetrics
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=avgmetric#pyspark.ml.tuning.CrossValidatorModel.avgMetrics
# the best f1 score is lower than the score obtained in the preliminary test

# extract Param Map of the best f1
cvModel_rf.extractParamMap()
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=extractparammap#pyspark.ml.classification.RandomForestClassifier.extractParamMap
# estimatorParamMaps is [{}]

# best RF model as the default model
rf_best = RandomForestClassifier()
# fit train set
rf_best_model = rf_best.fit(train)
# transfomr test set
results_final_rf = rf_best_model.transform(test)

# final resuls metrics from RF
# set evaluator 
evaluator = MulticlassClassificationEvaluator(predictionCol = 'prediction')
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator

# Print accuracy and f1 score
print("Test Set Metrics: ")
print('Accuracy: {}'.format(evaluator.evaluate(results_final_rf, {evaluator.metricName: 'accuracy'})))
print('F-1 Score: {}'.format(evaluator.evaluate(results_final_rf, {evaluator.metricName: 'f1'})))
# # http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=evaluate#pyspark.ml.evaluation.MulticlassClassificationEvaluator.evaluate
```

由于并未找到最优参数，故此仅以默认参数进行最终测试。

同时以``featureImportances``获得最优模型中的各输入特征值的重要性，以反映其对判断顾客流失的影响程度。



# 结果

### 模型的评价和验证

本项目使用了梯度提升树模型和随机森林模型对数据集进行了训练，调优，和最终测试。所得结果如下：

```
Test Set Metrics in GBT: 
Accuracy: 0.7941176470588235
F-1 Score: 0.7820069204152248
-----------------------------------------
Test Set Metrics in RF: 
Accuracy: 0.8235294117647058
F-1 Score: 0.7955182072829133
```

梯度提升树模型使用了maxDepth = 5和 maxIter = 10最为最佳模型的参数，所得准确率为0.7941，F1值为0.7820。随机森林模型则使用了默认参数，即maxDepth = 5, maxBins = 32, impurity = ‘gini’, numTress = 20, 所得准确率为0.8235，F1值为0.7955。

对比这两个模型，随机森林的结果无论是准确率还是反映查全率和查准率的F1值都比梯度提升树模型所得结果更为好。

此外这两个模型的``featureImportances``结果展示如下：

<img src="\C:\Users\Isaac\Pictures/comparison.png" alt="Alt text" style="zoom:200%;" />

从特征重要性对比可知，两种模型所要求对于分类的特征重要性基本一致，lefttime即用户生命周期对于是评判顾客流失与否起到最主要的作用，num_thumb_down，add_friends， avg_songs_played等效果依次降低，而性别的影响显然是最低的。具体模型分析来讲，随机森林认为num_songs和 num_thumb_up对于判断顾客流失的重要性来说没有梯度提升树模型所显示的那么重要，仅与total_listen的重要性差不多。但认为add_to_playlist几乎与avg_songs_played和add_friend一样重要，这一点梯度提升树模型并没有反映出来。

### 讨论结果

按测试结果，随机森林的结果无论是准确率还是反映查全率和查准率的F1值都比梯度提升树模型所得结果更为好，分别为准确率为0.8235，F1值为0.7955。这得益于随机森林模型的独特算法结构。随机森林采用的是bagging的思想，即通过在训练样本集中进行有放回的采样得到多个采样集，基于每个采样集训练出一个基学习器，再将基学习器结合起来共同实现分类或者回归。随机森林在对决策树进行bagging的基础上，在决策树的训练过程中引入了随机属性选择。传统决策树在选择划分属性的时候是在当前节点属性集合中选择最优属性，而随机森林则是对结点先随机选择包含k个属性的子集，再选择最优属性，k作为一个参数控制了随机性的引入程度。 总结起来包括2点：

1. 有放回的采集多个采样集，训练多个基分类器。
2. 每个基分类器随机选择一些属性而不是全部属性。

因此在目前数据量仅为225条的情况下，随机森林事实上在其算法中已经使用了类似交叉验证的方法增加了数据量，同时又平均多次试验结果，获得了较为可靠的评判结果。同时其优点还在于很好的避免了过拟合。

而梯度提升模型由多棵树组成的模型，但是又boosting结果构成的，最终结果都是由多棵树共同决定。但不同随机森林梯度提升树模型只能是回归树，对异常值很敏感，同时其本质是基于权值的弱分类器的集成，结果是加权累加所得。这不可避免得到了相对差一些的结果。虽然经过参数优化，准确率和F1值相比默认参数来讲已大为提高了近0.7-0.8，但依然不及随机森林的结果好。

另外有一点需要指出的就是，随机森林模型既可以处理离散数据也可以处理连续数据，且数据不需要归一化处理。同时还可以很好处理缺失数据（随机森林可以选择特定维度进行，所有某些维度去掉依然可行），并容易并行化。这也是为什么随机森林计算耗时仅为梯度提升树计算耗时一半的主要原因。这对于进一步进行12G全数据集运算是极有好处的。

对于目前随机森林模型参数仅为默认值的情况，事实上还可考虑进一步拓展参数网格的参数，可在目前现有maxDepth和numTrees的基础上再进一步增加maxBins，并增加参数范围以获得最佳参数。同时为了增加数据量以获得更好的结果，原来3-fold交叉验证也可增加至10-fold，观察其结果，以获得最优参数。

根据``featureImportances``所得特征重要性来看，随机森林模型认为lifetime最为重要，超过0.35以上。可见用户的生命周期是评判顾客是否流失的最为重要的因素。这说明用户使用Sparkify服务的长短可直接反映出其对此平台的使用适应程度和舒适程度，以及所耗费的时间成本。这些都是用户决定选择其他平台是需要考虑的成本。因此用户生命周期最为重要。

此外avg_songs_played连续听歌平均数目，add_friend添加朋友次数，add_to_playlist添加歌曲至歌单和num_thumb_down点差数的重要性均在0.10-0.13左右。这些都与用户在Sparkify上使用其功能有关。制作自己个性化的歌单，找到相同音乐偏好的朋友，发展为自己的人脉等都是使音乐平台从单一的听歌功能发展到个性空间，甚至是音乐社区相关的功能。用户在这些方面参与越多，某种程度上也就对音乐平台就愈发认可和依赖，甚至可以说成为用户生活生命的一部分。这样也就更加难以割舍，自然其顾客忠诚度就更高。有一点需要指出，通常来说num_thumb_down为对歌曲的点差，即以简单的点击形式表达用户自己对歌曲的好恶，这往往与用户体验有关。一般来说，点差越多，用户流失的可能性会比较高。相比重要性仅低于0.05的num_thumb_up点赞数来说，点差数更能明确反映出用户对于歌曲的偏好，甚至发展到对整个音乐平台的好恶，以至于选择离开。但反过来讲，如果Sparkify加倍分析点差歌曲的特点，差异化的确定用户的偏好，并以此为据提供差异化的服务，反而可以对于保持用户忠诚度提供极好的参考。

剩余的特征中只有num_songs听歌数目略高于0.05, 这是一个反映用户对Sparkify频率的特征。若是除以lifetime，可直接反映出每天使用Sparkify音乐平台听歌的数量。这个指标足以反映出用户对于平台的使用和依赖程度。因此很自然成为了评判顾客流失与否的特征。

考虑以上特征后，模型可去除其它三个低于0.05的特征，更新输入特征，重新进行数据训练和测试以期获得更好的实验结果。



# 结论

### 反思

本项目为Sparkify数据集分析并确定顾客流失的分类项目。项目仅选择了远小于原12G数据集的128M数据集。主要步骤如下：

1. 下载Spariky数据集。此步骤中还涉及加载Spark环境。
2. 探索性分析数据集，并确定与顾客流失相关的特征。此步骤中还涉及顾客流失明确的定义，可视化分析基于userId的多个与顾客流失相关的特征分析。
3. 对相关特征进行聚合并重组。此步骤主要为选择与顾客流失相关的特征选择，但同时还涉及到对这些特征的合并，数据转化等以适用于进一步的模型训练和验证。
4. 设计模型进行分类及进一步调优。此步骤主要涉及到基准于预测模型的确定和多个模型的设计训练和验证。并最终根据准确率和F1值及耗时成本，选择最终模型，当然预先考虑大数据集测试同样是选择的重要考虑项。
5. 分析分类指标及确定主要影响特征。此步骤主要涉及到对最优模型的测试结果的分析，以及对主要影响特征的进一步筛选。

在众多步骤中，步骤三，步骤四相对来讲是最为有挑战性的。对于特征选择来说，发现直接使用原数据集的特征并不能带来好的结果。数据集需要根据userId进一步分类聚合，获得适合进一步分析的特征数据集来。那么哪些特征是与预测顾客流失相关的，或是最为相关的，其实是个很难的问题。因为这完全是通过主观假设和经验来确定的，几乎无法求助于什么明确的公式。这必然与实际可能会存在偏差。而对于模型设计来说，同样的情况也会出现。哪种模型更适合目前数据集，只能选择从相对较简单的逻辑回归，到功能强大的支持向量机，以及两只集成方法，以考虑全数据分析的需要。但如何提高模型设计的效率依然是值得思考的问题。

此外使用Spark来分析数据集，构建机器学习模型，也是相对有挑战的。不同python的packages，如pandas, matplotlib，numpy和sklearn等，有着相对明确和详尽的documentaion和examples。Spark的packages的支持文件相对比较简单，很多参数的定义也过于简单，且例子较少这给项目分析和模型训练测试带来了很多困难。



### 改进

目前项目代码并没有使用Spark所带的``pipeline``功能而是多个模型均各自的分类器编写。这样的效率其实并不高。虽然目前的代码在模型训练和验证，各阶段代码很清晰，但是没有将这些阶段直接合并在pipeline中。若使用pipeline将多个阶段合并的化，代码将更为简洁，同时也可以在几个模型中复用，提高效率。这一点值得进一步全数据训练测试时使用。
