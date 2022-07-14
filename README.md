# Practical Time Series  Aileen Nielsen
Saving handy tips and code for timeseries analysis

https://learning-oreilly-com.proxy.lib.umich.edu/library/view/practical-time-series/9781492041641/ch02.html#idm45554610083352

```YearJoined.groupby('memberId').count().groupby('memberStats').count()```

WHAT IS A LOOKAHEAD?
The term lookahead is used in time series analysis to denote any knowledge of the future. You shouldn’t have such knowledge when designing, training, or evaluating a model. A lookahead is a way, through data, to find out something about the future earlier than you ought to know it.

A lookahead is any way that information about what will happen in the future might propagate back in time in your modeling and affect how your model behaves earlier in time. For example, when choosing hyperparameters for a model, you might test the model at various times in your data set, then choose the best model and start at the beginning of your data to test this model. This is problematic because you chose the model for one time knowing things that would happen at a subsequent time—a lookahead.

MultiIndex from cross product in Pandas : Panel Data
```complete_idx = pd.MultiIndex.from_product((set(emails.week), set(emails.member)))```

