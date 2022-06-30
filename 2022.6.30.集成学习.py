# ,
cv = KFold(n_splits=5,shuffle=True, random_state=0) 
result_t = cross_validate(DTR, #
X_train,Y_train, #
cv=cv, #
scoring="neg_mean_squared_error", #
return_train_score=True, #
verbose=True, #
n_jobs=4#
 )
result_f = cross_validate(RFR,X_train,Y_train,
cv=cv,
scoring="neg_mean_squared_error",
return_train_score=True,
verbose=True,
n_jobs=4)
