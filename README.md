# Practical Time Series  Aileen Nielsen
Saving handy tips and code for timeseries analysis

https://learning-oreilly-com.proxy.lib.umich.edu/library/view/practical-time-series/9781492041641/ch02.html#idm45554610083352

```YearJoined.groupby('memberId').count().groupby('memberStats').count()```

WHAT IS A LOOKAHEAD?
The term lookahead is used in time series analysis to denote any knowledge of the future. You shouldn’t have such knowledge when designing, training, or evaluating a model. A lookahead is a way, through data, to find out something about the future earlier than you ought to know it.

A lookahead is any way that information about what will happen in the future might propagate back in time in your modeling and affect how your model behaves earlier in time. For example, when choosing hyperparameters for a model, you might test the model at various times in your data set, then choose the best model and start at the beginning of your data to test this model. This is problematic because you chose the model for one time knowing things that would happen at a subsequent time—a lookahead.

MultiIndex from cross product in Pandas : Panel Data
```complete_idx = pd.MultiIndex.from_product((set(emails.week), set(emails.member)))```

**[Shift the target forward to predict](https://stackoverflow.com/questions/53335567/use-pandas-shift-within-a-group)**
```
df['target'] = df.amount.shift(1)

df['prev_value'] = df.groupby('object')['value'].shift()

Just beware, it is safer to sort dataframe beforehand: df.sort_values(by=['period']).groupby('object')['value'].shift()


```

### Summary 1
- Recalibrate the resolution of our data to suit our question. Often data comes with more specific time information than we need.
- Understand how we can avoid lookahead by not using data for timestamps that produce the data’s availability.
- Record all relevant time periods even if “nothing happened.” A zero count is just as informative as any other count.
- Avoid lookahead by not using data for timestamps that produce information we shouldn’t yet know about. (shift prediction)
- [This means that methods that randomize the dataset during evaluation, like k-fold cross-validation, cannot be used. Instead, we must use a technique called walk-forward validation](https://machinelearningmastery.com/xgboost-for-time-series-forecasting/)


##Chapter 8 features of time series

ean and variance

Maximum and minimum

Difference between last and first values

You will also visually identify other features that are more computationally challenging to compute but are often useful. Some examples include:

Number of local maxima and minima

Smoothness of the time series

Periodicity and autocorrelation of the time series


12h 2m remaining
Chapter 8. Generating and Selecting Features for a Time Series
In the previous two chapters we examined methods of time series analysis that rely on using all the data points in a time series to fit a model. However, in preparation for the next chapter’s discussion of the application of machine learning to time series analysis, in this chapter we will study feature generation and selection for time series. If you are unfamiliar with the concept of feature generation, you will not remain so for long. It’s an intuitive process and one that enables a creative side to data analysis.

Feature generation is the process of finding a quantitative way to encapsulate the most important traits of time series data into just a few numeric values and categorical labels. You are compressing the raw times series data into a shorter representation via a set of features to describe that time series (we’ll work through a quick example momentarily). For example, a very simple feature generation could describe every time series with its mean value and the number of time steps in the series. This would be one way of describing that time series without going through all the raw data step by step.

The purpose of feature generation is to compress as much information about the full time series as possible into a few metrics or, alternately, to use those metrics to identify the most important information about the time series and discard the rest. This is important for machine learning methods, most of which were developed on nontemporal data but which can be fruitfully applied to time series problems, provided we can digest a time series into a properly formatted input. In this chapter we will focus particularly on packages that allow us to automatically generate commonly used time series features so there will be no need for us to reinvent or handcode them.

Once we have generated some putatively useful features, we must ensure that they are indeed useful. While you are unlikely to craft too many unhelpful features by hand, this is a problem you will run into when you use code that automatically generates a large number of features of a time series for downstream use in machine learning. For this reason, we must inspect the features, once generated, to see which can be discarded in subsequent analyses.

Traditional machine learning models were not originally developed with time series in mind, and so they do not automatically lend themselves to time series analytical applications. However, one way to make these models work with temporal data is feature generation. For example, by describing a univariate time series not with a series of numbers detailing the step-by-step outputs of a process but rather by describing it with a set of features, we can access methods designed for cross-sectional data.

In this chapter we will first work through a very simple example of feature generation for a short time series. We will then review feature generation packages for time series, both in R and Python. Finally, we’ll work through an example of automated feature generation and feature selection. After reading this chapter you will have all the skills needed to preprocess a time series data set for downstream machine learning applications in Chapter 9.

Introductory Example
Imagine the past week’s morning, midday, and evening temperatures were as shown in Table 8-1.

Table 8-1. Temperatures for the past week
Time	Temperature (°F)
Monday morning	35
Monday midday	52
Monday evening	15
Tuesday morning	37
Tuesday midday	52
Tuesday evening	15
Wednesday morning	37
Wednesday midday	54
Wednesday evening	16
Thursday morning	39
Thursday midday	51
Thursday evening	12
Friday morning	41
Friday midday	55
Friday evening	20
Saturday morning	43
Saturday midday	58
Saturday evening	22
Sunday morning	46
Sunday midday	61
Sunday evening	35
You could plot this data and you’d see elements of periodicity (a daily cycle) and also a trend of overall increasing temperatures. But we can’t store an image of a plot in a database, and most methods that accept a picture as an input are data-intensive and seek to strip down the picture into summary metrics. So we should do the summary metrics ourselves. Instead of describing the 21 numbers in Table 8-1 as a time series, we could describe the series with a few words and numbers:

Daily/periodic

Increasing trend; we could make this more quantitative by computing a slope

Mean values for each of morning, midday, and evening

By doing so, we’d summarize the 21-point time series with 2 to 5 numbers—quite a bit of data compression without losing too much detail. This is a simple case of feature generation. Then, feature selection would entail paring away any features that were not descriptive enough to justify inclusion. What justifies inclusion will depend on our downstream use of the features.

General Considerations When Computing Features
As with any aspect of analysis, when you are computing time series features for a time series data set, you will want to think through whether your analysis makes sense and whether the effort you put into generating features is more likely to lead to overfitting from a surfeit of features than it is to lead to meaningful insights.

The best approach is to develop a set of potentially useful features as you run through time series exploration and cleaning. As you visualize data and think about what distinguishes different time series in the same data set or different time periods in the same time series, you will develop ideas about what kinds of measurements would be useful for labeling or predicting a time series. You can also draw useful assistance from any background knowledge you have about a system or even a working hypotheses you’d like to test with subsequent analysis.

Next we discuss a few distinct concerns you should keep in mind when generating time series features.

The Nature of the Time Series
As you decide what time series features to generate, you need to keep in mind the underlying attributes of your time series, which you determined during data exploration and cleaning.

Stationarity
Stationarity is one consideration. Many time series features assume stationarity and are useless unless the underlying data is stationary or at least ergodic. For example, using the mean of a time series as a feature is practical only where the time series is stationary so that the idea of a mean makes sense. This value is not very meaningful where we have a nonstationary time series, as the value measured as the mean in that case is more or less an accident, a result of too many entangled processes, such as a trend or a seasonal cycle.

STATIONARY VERSUS ERGODIC TIME SERIES
An ergodic time series is one in which every (reasonably large) subsample is equally representative of the series. This is a weaker label than stationarity, which requires that these subsamples have equal mean and variance. Ergodicity requires that each slice in time is “equal” in containing information about the time series, but not necessarily equal in its statistical measurements (mean and variance). You can find a helpful discussion on StackExchange.

Length of time series
Another consideration for feature generation is the length of the time series. Some features may be sensible for a stationary time series but become unstable as the length of the time series increases, such as the minimum and maximum value of the series. For the same underlying process, a longer time series will likely measure more extreme maximum and minimum values than a shorter time series produced by the same process, simply because there were more opportunities for data collection.

Domain Knowledge
Domain knowledge should be key for time series feature generation where you are lucky enough to have some insights. Some examples of how domain knowledge is applied to generate specific time series features are provided later in this chapter, but for now we’ll focus on the more general point.

For example, if you are working with a physics time series, you should quantify features that make sense on the timescale of the system you are studying, as well as make sure that the features you select would not be unduly influenced by the characteristics of, say, the error of a sensor rather than the characters of an underlying system.

As another example, imagine you are working with data from a specific financial market. To ensure financial stability, this market imposes maximum price changes in a given day. If the price changes too much, the market shuts down. You might consider whether, in this context, to generate a feature indicating the maximum price seen on a given day.

External Considerations
The extent of your computational resources and associated storage resources is also important. Likewise, your motivation for generating features matters. Are you generating features that will be stored so that you can throw out voluminous raw data? Or are you merely computing the features for a single analysis and planning to keep only the raw data?

The purpose of your time series feature generation may influence how many features you decide to compute and whether you should contemplate particularly computationally demanding features. This may also depend on the overall size of the data set you are analyzing. For a small data set all these decisions will be low stakes, but in the case of extremely large time series data sets, you may risk embarking on a feature generation task that will be left half-done, wasting computational energy and coding.

After considering all these factors, try putting together a list of features and running them on a small data set to get an idea of how fast or slow they run. If the small set runs too slowly, you should pare down your time series substantially before continuing your analysis. Likewise, you might consider exploring the usefulness of computationally taxing features on a subset of your data before undertaking an analysis with the full data set.

A Catalog of Places to Find Features for Inspiration
Time series feature generation is limited only by your data, your imagination, your coding skill, and your domain knowledge. So long as you can think of a reasonably general and well-defined way to quantify the behavior of a time series, you can generate a feature. Some simple and oft-used time series features amount to the same summary statistical functions you will have used in other applications, such as:

Mean and variance

Maximum and minimum

Difference between last and first values

You will also visually identify other features that are more computationally challenging to compute but are often useful. Some examples include:

Number of local maxima and minima

Smoothness of the time series

Periodicity and autocorrelation of the time series

In such cases, you will need to make some implementation definitions, as there are different ways to identify these commonly used features. It will help to keep your own personal library of feature generation code available, but you may also want to look into feature generation libraries for time series data, particularly as you become interested in the more computationally demanding features. In such cases, you should look for an excellent implementation such that the code is both reliable and efficient.

Now we’ll turn to the use of time series feature generation libraries, paying particular attention to the wide range of features you can benefit from via automatic feature generation.

Open Source Time Series Feature Generation Libraries
There have been many efforts to automate the creation of time series features because they tend to be interesting, descriptive, and even predictive across domains.

The tsfresh Python module
One particularly compelling example of automatic feature generation in Python is the tsfresh module, which implements a large and general set of features. We can get a sense of the breadth of implemented features by considering some of the general categories of features that are available. These include:

Descriptive statistics
These are driven by the traditional statistical time series methodologies we studied in Chapter 6, including:

An Augmented Dickey–Fuller test value

An AR(k) coefficient

The autocorrelation for a lag, k

Physics-inspired indicators of nonlinearity and complexity
This category includes:

The function c3(), which is a proxy for calculating the expected value of  ( is the lag operator). This has been proposed as a measure of nonlinearity in a time series.

The function cid_ce(), which calculates the square root of the sum from 0 to n – 2 × lag of (xi – xi + 1)2. This has been proposed as a measure of the complexity of a time series.

The function friedrich_coefficients(), returns coefficients of a model fitted to describe complex nonlinear motion.

History-compressing counts
This category comprises features such as:

The sum of the values in a time series that occur more than once

The length of the longest consecutive subsequence that is above or below the mean

The earliest occurrence within the time series of the minimum or maximum value
