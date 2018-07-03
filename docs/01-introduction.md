# Introduction



Machine learning continues to grow in importance for many organizations across nearly all domains. Examples include:

* predicting the likelihood of a patient returning to the hospital (_readmission_) within 30 days of discharge,
* segmenting customers based on common attributes or purchasing behavior for target marketing,
* predicting coupon redemption rates for a given marketing campaign,
* predicting customer churn so an organization can perform preventative intervention,
* and many more!

In essence, these tasks all seek to learn from data.  To address each scenario, we use a given set of _features_ to train an algorithm and extract insights. These algorithms, or _learners_, can be classified according to the amount and type of supervision provided during training.  The two main groups this book focuses on includes: ___supervised learners___ that are used to construct predictive models, and ___unsupervised learners___ that are used to build descriptive models. Which type you will need to use depends on the learning task you hope to accomplish.


## Supervised Learning

A ___predictive model___ is used for tasks that involve the prediction of a given output using other variables and their values (_features_) in the data set. Or as stated by @apm, predictive modeling is _"the process of developing a mathematical tool or model that generates an accurate prediction_ (p. 2).  The learning algorithm in a predictive attempts to discover and model the relationship among the ___target___ response (the variable being predicted) and the other features (aka predictor variables). Examples of predictive modeling include:

* using customer attributes to predict the probability of the customer churning in the next 6 weeks,
* using home attributes to predict the sales price,
* using employee attributes to predict the likelihood of attrition,
* using patient attributes and symptoms to predict the risk of readmission,
* using production attributes to predict time to market.

Each of these examples have a defined learning task.  They each inted to use attributes $X$ to predict an outcome measurement $Y$.

<div class="rmdnote">
<p>Throughout this text I will use various terms interchangeably for:</p>
<ul>
<li><span class="math inline">\(X\)</span>: &quot;predictor variables&quot;, &quot;independent variables&quot;, &quot;attributes&quot;, &quot;features&quot;, &quot;predictors&quot;</li>
<li><span class="math inline">\(Y\)</span>: &quot;target variable&quot;, &quot;dependent variable&quot;, &quot;response&quot;, &quot;outcome measurement&quot;</li>
</ul>
</div>

The predictive modeling examples above describe what is known as _supervised learning_.  The supervision refers to the fact that the target values provide a supervisory role, which indicates to the learner the task it needs to learn. Specifically, given a set of data, the learning algorithm attempts to optimize a function (the algorithmic steps) to find the combination of feature values that results in a predicted value that is as close to the actual target output as possible.

<div class="rmdnote">
<p>In supervised learning, the training data you feed to the algorithm includes the desired solutions. Consequently, the solutions can be used to help <em>supervise</em> the training process to find the optimal algorithm parameters.</p>
</div>

Supervised learning problems revolve around two primary themes: regression and classification.

### Regression problems

When the objective of our supervised learning is to predict a numeric outcome, we refer to this as a ___regression problem___ (not to be confused with linear regression modeling).  Regression problems revolve around predicting output that falls on a continuous numeric spectrum. In the examples above predicting home sales prices and time to market reflect a regression problem because the output is numeric and continuous.  This means, given the combination of predictor values, the response value could fall anywhere along the continuous spectrum.  The following illustrates average home sales prices as a function of two home features: year built and total square footage. Depending on the combination of these two features, the expected home sales price could fall anywhere along the plane.

<div class="figure" style="text-align: center">
<!--html_preserve--><div id="3a64495f5373" style="width:672px;height:480px;" class="plotly html-widget"></div>
<script type="application/json" data-for="3a64495f5373">{"x":{"visdat":{"3a645bd629ff":["function () ","plotlyVisDat"]},"cur_data":"3a645bd629ff","attrs":{"3a645bd629ff":{"x":[334,882,988,1082,1177,1268,1358,1442,1520,1614,1696,1808,1982,2315,5642],"y":[1872,1920,1935,1949,1956,1962,1967,1973,1979,1993,1998,2003,2005,2006,2010],"z":[[148582.368746955,153868.347564933,162109.337964625,162241.48962845,162307.209908747,162363.354651755,162410.0109362,162465.842180157,162521.503895785,162650.727703662,162696.659184125,162742.475864468,162760.770520939,162769.911006924,154666.106069741],[167295.480289638,173265.344633435,182551.06771714,182705.302137827,182782.004383304,182847.531003718,182901.983628788,182967.144363953,183032.107242329,183182.924557601,183236.531262773,183290.003984538,183311.355707708,183322.023583691,174196.409875262],[169289.572900254,175333.072739417,184730.409672929,184887.224994192,184965.210746092,185031.833864237,185087.19767898,185153.44878929,185219.498731951,185372.839767247,185427.343506788,185481.711020894,185503.42003574,185514.266423932,176279.718075545],[170844.522308742,176945.622532147,186430.061591219,186588.943708027,186667.957297024,186735.458494956,186791.551994049,186858.676281154,186925.59674926,187080.958789685,187136.180877951,187191.264945357,187213.260080811,187224.249422299,177904.744470357],[172247.792216477,178401.036431262,187964.142187936,188124.938141149,188204.903499208,188273.217792109,188329.986973165,188397.919815109,188465.646382924,188622.87985819,188678.76713172,188734.514721841,188756.774802727,188767.89651784,179371.711604784],[173457.667288051,179656.014275681,189286.999693937,189449.489222024,189530.296810988,189599.330620566,189656.69772002,189725.346061029,189793.785955338,189952.675483867,190009.15138715,190065.486135816,190087.980669827,190099.219523816,180636.913034135],[174543.130615764,180782.075839113,190474.011017404,190638.059951257,190719.64304755,190789.339371951,190847.257021256,190916.56417776,190985.660887108,191146.075272345,191203.093172705,191259.968563792,191282.678976983,191294.025689822,181772.388245311],[175469.140599147,181742.835473086,191486.81031574,191652.224124383,191734.485985901,191804.762178478,191863.16169827,191933.045485102,192002.717073875,192164.466094732,192221.958379734,192279.306969797,192302.206331917,192313.647448553,182741.387204016],[176262.27539834,182565.830235198,192354.413803149,192521.025596254,192603.883226848,192674.668384108,192733.490853579,192803.880763156,192874.05693786,193036.977401508,193094.886065678,193152.649994221,193175.715201504,193187.239178695,183571.613825837],[177142.006037372,183478.800446524,193316.908132462,193484.884509846,193568.420761191,193639.785663204,193699.089900595,193770.056317766,193840.807249533,194005.062064725,194063.445012566,194121.682039365,194144.936155359,194156.554516255,184492.821608167],[177848.465079361,184212.056171444,194089.972826159,194259.076182861,194343.172891748,194415.016591851,194474.71871048,194546.161252244,194617.38686288,194782.743688858,194841.518336863,194900.146084821,194923.556216251,194935.252526514,185232.880556819],[178731.627753711,185128.869531836,195056.609383331,195227.167003552,195311.98693153,195384.448476684,195444.664024585,195516.720961499,195588.55910171,195755.337971605,195814.618072774,195873.750010575,195897.361465627,195909.158362406,186158.472853909],[179940.474980516,186384.107181103,196380.167839301,196552.815028272,196638.674118678,196712.023418834,196772.976691187,196845.916426088,196918.634683719,197087.456827401,197147.463192495,197207.319579015,197231.220307326,197243.161732545,187426.324580009],[181804.681259652,188320.88592864,198422.694812959,198598.873766711,198686.48923736,198761.339010686,198823.539176325,198897.971006106,198972.17682796,199144.452489217,199205.686377147,199266.767218458,199291.156873065,199303.342578571,189384.423493407],[185980.180437336,192700.131459099,203054.29569757,203250.733069552,203348.423244415,203431.879823379,203501.232244963,203584.222822002,203666.961402974,203859.046651712,203927.321685505,203995.426074198,204022.620239859,204036.207151998,193885.962795802]],"showscale":false,"alpha":1,"sizes":[10,100],"type":"surface"}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"scene":{"xaxis":{"title":"Feature: square footage"},"yaxis":{"title":"Feature: year built"},"zaxis":{"title":"Response: sale price"}},"xaxis":{"domain":[0,1]},"yaxis":{"domain":[0,1]},"hovermode":"closest","showlegend":false},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"cloud":false},"data":[{"colorbar":{"title":"","ticklen":2},"colorscale":[["0","rgba(68,1,84,1)"],["0.252141423454362","rgba(60,82,138,1)"],["0.43899200574149","rgba(39,129,142,1)"],["0.515016419274956","rgba(36,148,139,1)"],["0.58643299073279","rgba(39,165,133,1)"],["0.61855887695757","rgba(46,172,128,1)"],["0.630961644590138","rgba(48,175,126,1)"],["0.659883927648207","rgba(52,181,122,1)"],["0.680323095331825","rgba(63,186,117,1)"],["0.695241542152464","rgba(72,189,113,1)"],["0.716660054673134","rgba(83,193,108,1)"],["0.735198524019955","rgba(92,196,102,1)"],["0.748065753101924","rgba(97,199,98,1)"],["0.765623531626216","rgba(104,203,93,1)"],["0.780132226893482","rgba(111,205,88,1)"],["0.792346537463537","rgba(120,207,84,1)"],["0.803875275938328","rgba(128,209,81,1)"],["0.8166890426826","rgba(137,211,76,1)"],["0.826500908920441","rgba(143,213,73,1)"],["0.839095985323491","rgba(151,214,68,1)"],["0.85309013035521","rgba(159,217,62,1)"],["0.874693068605234","rgba(172,220,52,1)"],["0.906447280928449","rgba(192,224,43,1)"],["0.987009965046187","rgba(245,230,38,1)"],["1","rgba(253,231,37,1)"]],"showscale":false,"x":[334,882,988,1082,1177,1268,1358,1442,1520,1614,1696,1808,1982,2315,5642],"y":[1872,1920,1935,1949,1956,1962,1967,1973,1979,1993,1998,2003,2005,2006,2010],"z":[[148582.368746955,153868.347564933,162109.337964625,162241.48962845,162307.209908747,162363.354651755,162410.0109362,162465.842180157,162521.503895785,162650.727703662,162696.659184125,162742.475864468,162760.770520939,162769.911006924,154666.106069741],[167295.480289638,173265.344633435,182551.06771714,182705.302137827,182782.004383304,182847.531003718,182901.983628788,182967.144363953,183032.107242329,183182.924557601,183236.531262773,183290.003984538,183311.355707708,183322.023583691,174196.409875262],[169289.572900254,175333.072739417,184730.409672929,184887.224994192,184965.210746092,185031.833864237,185087.19767898,185153.44878929,185219.498731951,185372.839767247,185427.343506788,185481.711020894,185503.42003574,185514.266423932,176279.718075545],[170844.522308742,176945.622532147,186430.061591219,186588.943708027,186667.957297024,186735.458494956,186791.551994049,186858.676281154,186925.59674926,187080.958789685,187136.180877951,187191.264945357,187213.260080811,187224.249422299,177904.744470357],[172247.792216477,178401.036431262,187964.142187936,188124.938141149,188204.903499208,188273.217792109,188329.986973165,188397.919815109,188465.646382924,188622.87985819,188678.76713172,188734.514721841,188756.774802727,188767.89651784,179371.711604784],[173457.667288051,179656.014275681,189286.999693937,189449.489222024,189530.296810988,189599.330620566,189656.69772002,189725.346061029,189793.785955338,189952.675483867,190009.15138715,190065.486135816,190087.980669827,190099.219523816,180636.913034135],[174543.130615764,180782.075839113,190474.011017404,190638.059951257,190719.64304755,190789.339371951,190847.257021256,190916.56417776,190985.660887108,191146.075272345,191203.093172705,191259.968563792,191282.678976983,191294.025689822,181772.388245311],[175469.140599147,181742.835473086,191486.81031574,191652.224124383,191734.485985901,191804.762178478,191863.16169827,191933.045485102,192002.717073875,192164.466094732,192221.958379734,192279.306969797,192302.206331917,192313.647448553,182741.387204016],[176262.27539834,182565.830235198,192354.413803149,192521.025596254,192603.883226848,192674.668384108,192733.490853579,192803.880763156,192874.05693786,193036.977401508,193094.886065678,193152.649994221,193175.715201504,193187.239178695,183571.613825837],[177142.006037372,183478.800446524,193316.908132462,193484.884509846,193568.420761191,193639.785663204,193699.089900595,193770.056317766,193840.807249533,194005.062064725,194063.445012566,194121.682039365,194144.936155359,194156.554516255,184492.821608167],[177848.465079361,184212.056171444,194089.972826159,194259.076182861,194343.172891748,194415.016591851,194474.71871048,194546.161252244,194617.38686288,194782.743688858,194841.518336863,194900.146084821,194923.556216251,194935.252526514,185232.880556819],[178731.627753711,185128.869531836,195056.609383331,195227.167003552,195311.98693153,195384.448476684,195444.664024585,195516.720961499,195588.55910171,195755.337971605,195814.618072774,195873.750010575,195897.361465627,195909.158362406,186158.472853909],[179940.474980516,186384.107181103,196380.167839301,196552.815028272,196638.674118678,196712.023418834,196772.976691187,196845.916426088,196918.634683719,197087.456827401,197147.463192495,197207.319579015,197231.220307326,197243.161732545,187426.324580009],[181804.681259652,188320.88592864,198422.694812959,198598.873766711,198686.48923736,198761.339010686,198823.539176325,198897.971006106,198972.17682796,199144.452489217,199205.686377147,199266.767218458,199291.156873065,199303.342578571,189384.423493407],[185980.180437336,192700.131459099,203054.29569757,203250.733069552,203348.423244415,203431.879823379,203501.232244963,203584.222822002,203666.961402974,203859.046651712,203927.321685505,203995.426074198,204022.620239859,204036.207151998,193885.962795802]],"type":"surface","frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1}},"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":{"render":[{"code":"function(el, x) { var ctConfig = crosstalk.var('plotlyCrosstalkOpts').set({\"on\":\"plotly_click\",\"persistent\":false,\"dynamic\":false,\"selectize\":false,\"opacityDim\":0.2,\"selected\":{\"opacity\":1}}); }","data":null}]}}</script><!--/html_preserve-->
<p class="caption">(\#fig:regression-problem)Average home sales price as a function of year built and total square footage.</p>
</div>


### Classification problems

When the objective of our supervised learning is to predict a categorical response, we refer to this as a ___classification problem___.  Classification problems most commonly revolve around predicting a binary or multinomial response measure such as:

* did a customer redeem a coupon (yes/no, 1/0),
* did a customer churn (yes/no, 1/0),
* did a customer click on our online ad (yes/no, 1/0),
* classifying customer reviews:
    * binary: positive vs negative
    * multinomial: extremely negative to extremely positive on a 0-5 Likert scale
    




However, when we apply machine learning models for classification problems, rather than predict a particular class (i.e. "yes" or "no"), we often predict the _probability_ of a particular class (i.e. yes: .65, no: .35).  Then the class with the highest probability becomes the predicted class.  Consequently, even though we are performing a classification problem, we are still predicting a numeric output (probability).  However, the essence of the problem still makes is a classification problem.


### Algorithm Comparison Guide

__TODO: keep this here or move reference guide to back???__

Although there are machine learning algorithms that can be applied to regression problems but not classification and vice versa, the supervised learning algorithms I cover in this both can be applied to both.  These algorithms have become the most popular machine learning applications in recent years. 

Although the chapters that follow will go into detail on each algorithm, the following provides a quick reference guide that compares and contrasts some of their features.  Moreover, I provide recommended base learner packages that I have found to scale well with typical rectangular data analyzed by organizations.


<table style="font-size:13px;">
<col width="40%">
<col width="15%">
<col width="15%">
<col width="15%">
<col width="15%">
<thead>
<tr class="header">
<th align="left">Characteristics</th>
<th align="left">Regularized GLM</th>
<th align="left">Random Forest</th>
<th align="left">Gradient Boosting Machine</th>
<th align="left">Deep Learning</th>
</tr>
</thead>
<tbody>

<tr class="odd">
<td align="left" valign="top">
  Allows n < p
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
</tr>

<tr class="even">
<td align="left" valign="top">
  Provides automatic feature selection
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
</tr>

<tr class="odd">
<td align="left" valign="top">
  Handles missing values
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="red" stroke-width="3" fill="red" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="red" stroke-width="3" fill="red" /></svg>
</td>
</tr>

<tr class="even">
<td align="left" valign="top">
  No feature pre-processing required
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="red" stroke-width="3" fill="red" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="red" stroke-width="3" fill="red" /></svg>
</td>
</tr>

<tr class="odd">
<td align="left" valign="top">
  Robust to outliers
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="red" stroke-width="3" fill="red" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="yellow" stroke-width="3" fill="yellow" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="red" stroke-width="3" fill="red" /></svg>
</td>
</tr>

<tr class="even">
<td align="left" valign="top">
  Easy to tune
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="red" stroke-width="3" fill="red" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="red" stroke-width="3" fill="red" /></svg>
</td>
</tr>

<tr class="odd">
<td align="left" valign="top">
  Computational speed
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="yellow" stroke-width="3" fill="yellow" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="yellow" stroke-width="3" fill="yellow" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="red" stroke-width="3" fill="red" /></svg>
</td>
</tr>

<tr class="even">
<td align="left" valign="top">
  Predictive power
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="red" stroke-width="3" fill="red" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="yellow" stroke-width="3" fill="yellow" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="green" stroke-width="3" fill="green" /></svg>
</td>
<td align="left" valign="center"> 
  <svg height="10" width="10"><circle cx="5" cy="5" r="5" stroke="yellow" stroke-width="3" fill="yellow" /></svg>
</td>
</tr>

<tr class="odd">
<td align="left" valign="top">
  Preferred regression base learner <br>
</td>
<td align="left" valign="top"> 
  <a  href="">glmnet</a><br>
  <a  href="">h2o.glm</a>
</td>
<td align="left" valign="top"> 
  <a  href="">ranger</a><br>
  <a  href="">h2o.randomForest</a>
</td>
<td align="left" valign="top"> 
  <a  href="">xgboost</a><br>
  <a  href="">h2o.gbm</a>
</td>
<td align="left" valign="top"> 
  <a  href="">h2o.deeplearning</a><br>
</td>
</tr>

<tr class="even">
<td align="left" valign="top">
  Preferred classifciation base learner <br>
</td>
<td align="left" valign="top"> 
  <a  href="">glmnet</a><br>
  <a  href="">h2o.glm</a>
</td>
<td align="left" valign="top"> 
  <a  href="">ranger</a><br>
  <a  href="">h2o.randomForest</a>
</td>
<td align="left" valign="top"> 
  <a  href="">xgboost</a><br>
  <a  href="">h2o.gbm</a>
</td>
<td align="left" valign="top"> 
  <a  href="">keras</a><br>
  <a  href="">h2o.deeplearning</a>
</td>
</tr>

</tbody>
</table>

## Unsupervised Learning

___Unsupervised learning___, in contrast to supervised learning, includes a set of statistical tools to better understand and describe your data but performs the analysis without a target variable.  In essence, unsupervised learning is concerned with identifying groups in a data set. The groups may be defined by the rows (i.e., *clustering*) or the columns (i.e., *dimension reduction*); however, the motive in each case is quite different.

The goal of ___clustering___ is to segment observations into similar groups based on the observed variables. For example, to divide consumers into different homogeneous groups, a process known as market segmentation.  In __dimension reduction__, we are often concerned with reducing the number of variables in a data set. For example, classical regression models break down in the presence of highly correlated features.  Dimension reduction techniques provide a method to reduce the feature set to a potentially smaller set of uncorrelated variables. These variables are often used as the input variables to downstream supervised models like.

Unsupervised learning is often performed as part of an exploratory data analysis. However, the exercise tends to be more subjective, and there is no simple goal for the analysis, such as prediction of a response. Furthermore, it can be hard to assess the quality of results obtained from unsupervised learning methods. The reason for this is simple. If we fit a predictive model using a supervised learning technique (i.e. linear regression), then it is possible to check our work by seeing how well our model predicts the response *Y* on observations not used in fitting the model. However, in unsupervised learning, there is no way to check our work because we don’t know the true answer—the problem is unsupervised.  

However, the importance of unsupervised learning should not be overlooked and techniques for unsupervised learning are used in organizations to: 

- Divide consumers into different homogeneous groups so that tailored marketing strategies can be developed and deployed for each segment.
- Identify groups of online shoppers with similar browsing and purchase histories, as well as items that are of particular interest to the shoppers within each group. Then an individual shopper can be preferentially shown the items in which he or she is particularly likely to be interested, based on the purchase histories of similar shoppers.
- Identify products that have similar purchasing behavior so that managers can manage them as product groups.

These questions, and many more, can be addressed with unsupervised learning.  Moreover, often the results of an unsupervised model can be used as inputs to downstream supervised learning models.


### Algorithm Decision Guide

TBD


## Machine learning interpretability

In his seminal 2001 paper, Leo Breiman popularized the phrase: _“the multiplicity of good models.”_ The phrase means that for the same set of input variables and prediction targets, complex machine learning algorithms can produce multiple accurate models with very similar, but not the exact same, internal architectures. 

Figure \@ref(fig:error-surface) is a depiction of a non-convex error surface that is representative of the error function for a machine learning algorithm with two inputs — say, a customer’s income and a customer’s age, and an output, such as the same customer’s probability of redeeming a coupon. This non-convex error surface with no obvious global minimum implies there are many different ways complex machine learning algorithms could learn to weigh a customer’s income and age to make a good decision about if they are likely to redeem a coupon. Each of these different weightings would create a different function for making coupon redemption (and therefore marketing) decisions, and each of these different functions would have different explanations.


<div class="figure" style="text-align: center">
<!--html_preserve--><div id="3a641f6ada8c" style="width:672px;height:480px;" class="plotly html-widget"></div>
<script type="application/json" data-for="3a641f6ada8c">{"x":{"visdat":{"3a64184be7fa":["function () ","plotlyVisDat"]},"cur_data":"3a64184be7fa","attrs":{"3a64184be7fa":{"showscale":false,"alpha":1,"sizes":[10,100],"z":{},"type":"surface","x":[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90],"y":[50,75,100,150,200,250]}},"layout":{"margin":{"b":40,"l":60,"t":25,"r":10},"scene":{"xaxis":{"title":"age"},"yaxis":{"title":"income"},"zaxis":{"title":"error"}},"xaxis":{"domain":[0,1]},"yaxis":{"domain":[0,1]},"hovermode":"closest","showlegend":false},"source":"A","config":{"modeBarButtonsToAdd":[{"name":"Collaborate","icon":{"width":1000,"ascent":500,"descent":-50,"path":"M487 375c7-10 9-23 5-36l-79-259c-3-12-11-23-22-31-11-8-22-12-35-12l-263 0c-15 0-29 5-43 15-13 10-23 23-28 37-5 13-5 25-1 37 0 0 0 3 1 7 1 5 1 8 1 11 0 2 0 4-1 6 0 3-1 5-1 6 1 2 2 4 3 6 1 2 2 4 4 6 2 3 4 5 5 7 5 7 9 16 13 26 4 10 7 19 9 26 0 2 0 5 0 9-1 4-1 6 0 8 0 2 2 5 4 8 3 3 5 5 5 7 4 6 8 15 12 26 4 11 7 19 7 26 1 1 0 4 0 9-1 4-1 7 0 8 1 2 3 5 6 8 4 4 6 6 6 7 4 5 8 13 13 24 4 11 7 20 7 28 1 1 0 4 0 7-1 3-1 6-1 7 0 2 1 4 3 6 1 1 3 4 5 6 2 3 3 5 5 6 1 2 3 5 4 9 2 3 3 7 5 10 1 3 2 6 4 10 2 4 4 7 6 9 2 3 4 5 7 7 3 2 7 3 11 3 3 0 8 0 13-1l0-1c7 2 12 2 14 2l218 0c14 0 25-5 32-16 8-10 10-23 6-37l-79-259c-7-22-13-37-20-43-7-7-19-10-37-10l-248 0c-5 0-9-2-11-5-2-3-2-7 0-12 4-13 18-20 41-20l264 0c5 0 10 2 16 5 5 3 8 6 10 11l85 282c2 5 2 10 2 17 7-3 13-7 17-13z m-304 0c-1-3-1-5 0-7 1-1 3-2 6-2l174 0c2 0 4 1 7 2 2 2 4 4 5 7l6 18c0 3 0 5-1 7-1 1-3 2-6 2l-173 0c-3 0-5-1-8-2-2-2-4-4-4-7z m-24-73c-1-3-1-5 0-7 2-2 3-2 6-2l174 0c2 0 5 0 7 2 3 2 4 4 5 7l6 18c1 2 0 5-1 6-1 2-3 3-5 3l-174 0c-3 0-5-1-7-3-3-1-4-4-5-6z"},"click":"function(gd) { \n        // is this being viewed in RStudio?\n        if (location.search == '?viewer_pane=1') {\n          alert('To learn about plotly for collaboration, visit:\\n https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html');\n        } else {\n          window.open('https://cpsievert.github.io/plotly_book/plot-ly-for-collaboration.html', '_blank');\n        }\n      }"}],"cloud":false},"data":[{"colorbar":{"title":"error","ticklen":2},"colorscale":[["0","rgba(68,1,84,1)"],["0.0555555555555545","rgba(71,23,102,1)"],["0.0925925925925908","rgba(72,35,114,1)"],["0.111111111111112","rgba(72,40,120,1)"],["0.12962962962963","rgba(71,46,123,1)"],["0.176697530864198","rgba(67,61,130,1)"],["0.240740740740739","rgba(61,79,138,1)"],["0.277777777777779","rgba(57,89,140,1)"],["0.296296296296297","rgba(55,94,140,1)"],["0.314814814814815","rgba(52,99,141,1)"],["0.333333333333333","rgba(49,104,142,1)"],["0.351851851851851","rgba(48,108,142,1)"],["0.37037037037037","rgba(47,113,142,1)"],["0.387345679012343","rgba(45,117,142,1)"],["0.418981481481483","rgba(42,124,142,1)"],["0.432098765432101","rgba(40,127,142,1)"],["0.462962962962964","rgba(38,135,141,1)"],["0.481481481481482","rgba(37,139,140,1)"],["0.599537037037036","rgba(42,168,131,1)"],["0.685185185185185","rgba(66,187,116,1)"],["0.803240740740738","rgba(128,209,81,1)"],["0.844135802469136","rgba(154,215,66,1)"],["0.868055555555557","rgba(168,219,55,1)"],["1","rgba(253,231,37,1)"]],"showscale":false,"z":[[8.83,8.92,8.8,8.91,8.97,9.2],[8.89,8.93,8.82,8.95,8.97,9.23],[8.81,8.91,8.78,8.94,8.91,9.2],[8.87,8.79,8.91,8.74,9.09,8.99],[8.9,8.85,8.94,8.81,9.11,8.99],[8.87,8.79,8.92,8.76,9.11,8.98],[8.89,8.9,8.75,8.93,9.04,9.18],[8.94,8.94,8.78,8.98,9.08,9.2],[8.85,8.92,8.77,8.99,9.05,9.19],[8.94,8.79,8.91,8.89,9.25,8.93],[8.96,8.88,8.95,8.99,9.28,8.97],[8.92,8.81,8.92,8.92,9.27,8.97],[8.84,8.9,8.8,9.1,9,9.18],[8.9,8.95,8.8,9.13,9.01,9.2],[8.82,8.92,8.77,9.11,9,9.18]],"type":"surface","x":[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90],"y":[50,75,100,150,200,250],"frame":null}],"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1}},"base_url":"https://plot.ly"},"evals":["config.modeBarButtonsToAdd.0.click"],"jsHooks":{"render":[{"code":"function(el, x) { var ctConfig = crosstalk.var('plotlyCrosstalkOpts').set({\"on\":\"plotly_click\",\"persistent\":false,\"dynamic\":false,\"selectize\":false,\"opacityDim\":0.2,\"selected\":{\"opacity\":1}}); }","data":null}]}}</script><!--/html_preserve-->
<p class="caption">(\#fig:error-surface)Non-convex error surface with many local minimas.</p>
</div>



All of this is an obstacle to data scientists.  On one hand, different models can have widely different predictions based on the same feature set.  Even models built from the same algorithm but with different hyperparameters can lead to different results. Consequently, practitioners should understand how different implementations of algorithms differ, which can cause variance in their results (i.e. a default `xgboost` model can produce very different results from a default `gbm` model, even though they both implement gradient boosting machines).  Alternatively, data scientists can experience very similar predictions from different models based on the same feature set. However, these models will have very different logic and structure leading to different interpretations.  Consequently, practitioners should understand how to interpret different types of models.

This book will provide you with a fundamental understanding to compare and contrast models and even package implementations of similiar algorithms.  Several machine learning interpretability techniques will be demonstrated to help you understand what is driving model and prediction performance.  This will allow you to be more effective and efficient in applying and understanding mutliple good models. 


## Example data sets



