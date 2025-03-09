# Statistical Rethinking 2023 (Part-1: L01~L10)
#study/statistical
## Refers
- ~[Statistical Rethinking 2023 Playlist](https://www.youtube.com/watch?v=FdnMWdICdRs&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus)~
- https://github.com/rmcelreath/stat_rethinking_2023?tab=readme-ov-file
- [[Statistical Rethinking 2023 (Part-2: L11~L20)]]
---
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-21%2020.23.48.png)
## L-01 The Golem of Prague
- 2024-02-20
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2019.51.46.png)
- Bayes vs. Frequentism —> No
- Causal inference —> Yes
### Science Before statistics
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2019.54.51.png)
### What is Causal Inference?
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2019.55.43.png)
### Causal prediction
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2019.57.09.png)
- *Consequences of an intervention*
### Causal imputation
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2019.58.33.png)
- *Counterfactual outcomes* 反事实
- Casual inference ==> Description ==> Design
### Causes are not optional
- Descriptive 描述性的 —> Causal model
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2020.00.30.png)
### DAGs: Directed Acyclic Graphs
- No strong assumptions
- Only relationship
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2020.02.40.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2020.10.18.png)
- intuition pumps
### Statistical Models
- Incredibly limiting
- Focus on rejecting null hypotheses
- Relationship between research and test not clear
- Industrial framework
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2020.15.25.png)
### Null models rarely unique
- Null population dynamics?
- Null phylogeny?
- Null ecological community?
- Null social network?
- Problem: Many processes produce similar distributions
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2020.23.52.png)
- Network permutation models: low power, high false postitives
### Hypotheses and Models
- Generative causal models
- Statistical models justified by generative models & questions —> estimands
- An effective way to produce estimates.
### Justifying “controls”
- Adjustment set
### Finite data, infinites problems.
- DAG is not enough 
- Need generative model to design debug inference
- Need a strategy to derive estimate and uncertainty
- Easiest approach: Bayesian data analysis
### Bayes is practical, not philosophical
- Simple analyses
  - Little difference, adds mess
- Realistic analyses
  - huge difference
- Measurement error, missing data, latent variables, regularization
- Bayesian models are generative
### Statistics wars are over
- Bayes no longer controversial or marginalized
- Bayesian method routine
- Waiting for teaching to catch up
- The action is in machine learning, which has different battles
### Owls
- Scientific data analyses
  - Amateur software engineering
- Three modes
  - Understand what you are doing
  - Document your work, reduce error
  - Respectable scientific workflow
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2020.50.37.png)
1. Theoretical estimand
2. Scientific (causal) models
3. Use 1 & 2 to build statistical models
4. Simulate from 2 to validate 3 yields 1
5. Analyze real data
### Dags, Golems & Owls
- DAGs
  - Transparent scientific assumptions to 
    - justify scientific effort
    - expose it to useful critique
    - connect theories to golems
- Golems
  - Brainless, powerful statistical models
- Owls
  - Documented procedures, quality assurance
- Theory —>  Model —> Evidence
---
## L-02 The Garden of Forking Data
### Workflow
1. Define generative model of the sample
2. Define a specific estimand
3. Design a statistical way to produce estimate
4. Test 3 using 1
5. Analyze sample, summarize
### Generative model of the globe
- Begin conceptually
  - How do the variables influence one another?¥
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2021.14.02.png)<!-- {"width":292} -->
- Generative assumptions:
  - What do the arrows mean exactly?
  - Function?
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2021.19.27.png)<!-- {"width":510} -->
- Bayesian data analysis
  - For each possible explanation of the sample,
  - Count all the ways the sample could happen.
  - Explanations with more ways to produce the sample are more plausible.
- Probability
  - Non-negative values that sum to 1
  - Better convert to probability
- Test before you Estimate
  1. Code a generative simulation
  2. Code an estimator
  3. Test 2 with 1
- Generative simulation
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2021.46.40.png)<!-- {"width":759} -->
- Test the simulation on extreme settings
  - Check bugs
  - **If you test nothing, you miss everything**
- Code the estimator
  1. Test the estimator where the answer is known
  2. Explore different sampling designs
  3. Develop intuition for sampling and estimation
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2021.49.52.png)<!-- {"width":759} -->
- More possibilities
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2021.55.53.png)
- Infinite possibilities
  - Beta distribution
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2021.59.22.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2022.03.46.png)
1. No minimum sample size
2. Shape embodies sample size
3. No point estimate
   - Distribution: mode or mean, max mode
   - Always use the entire distribution
4. No one true interval
   - Interval is not important
   - Intervals communicate shape of posterior
   - 95% is obvious superstition. 
   - Nothing magical happens at the boundary.
   - Because interval is arbitrary.
### Letters From My reviews
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2022.13.30.png)
### Analyze sample, summarize
- From Posterior to Prediction
- Implications of model depend upon entire posterior
- Must average any inference over entire posterior
- This usually requires integral calculus
- Or we can just take samples from the posterior
### Uncertainty —> Causal model —> Implications
- MCMC
  - Posterior distribution —> Predictive distribution for p —> Posterior predictive
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2022.25.05.png)
### Sampling is Fun and easy
- Sample from posterior, compute desired quantity for each sample, profit
- Much easier than doing integrals
- Turn a *calculus problem* into a *data summary* problem
- MCMC produces only samples anyway
### Sample is handsome and handy
- Model based forecasts
- Causal effects
- Counterfactuals
- Prior predictions
### Bayesian data analysis
- For each possible explanation of the data,
- Count all the ways data can happen.
- Explanation with more ways to produce the data are more plausible.
### Bayesian modesty
- No guarantee except logical
- Probability throes is a method of logically deducing implications of data under assumptions that you must choose.
- Any framework selling you more is hiding assumptions.
## Misclassification simulation
- unobserved variable p
- population size unobserved
- true samples unobserved
  - misclassified samples
  - measurement process
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-20%2023.06.00.png)
### Measurement matters
- When there is measurement error, better to model it than to ignore it.
- Some goes for: 
  - missing data, compliance, inclusion, etc..
- Good news: 
  - Samples do not need to be representative of population in order to provide good estimates of population.
- What matters is why the sample differs
---
## L-03 Geocentric Models
### Linear Regression
Many special cases
- ANOVA, ANCOVA, t-test, others
- External causal model
### Gaussian Distribution
Why normal
- Generative
  - Summed fluctuations tend towards normal distribution
- Inferential
  - For estimating mean and variance, normal distribution is least informative distribution (Maxent)
- Variable does not have to be normal distributed for normal model to be useful.
- It is a machine for estimating mean / variance.
Making Geocentric Models
- Skill development goals
  - Language for representing models
  - Calculate posterior distributions with multiple unknowns
  - Constructing and understanding linear models
- Owl-drawing workflow
  1. State a clear *question*
  2. Sketch your causal *assumptions*
  3. Use the sketch to define a *generative* model
  4. Use generative model to build *estimator*
  5. Profit
### Linear Regression
Drawing the Owl
1. Question goal estimated (DAGs)
   - Describe association between *Adult* weight and height.
2. Scientific model
   - How does height influence weight?
   - H —> W; W = f(H)
   - *Weight is some function of height*
3. Generative models
   - Dynamic (Detail model)
     - Incremental growth of organism; 
     - both mass and height (length) derive from growth pattern;
     - Gaussian variation result of summed fluctuations
   - Static
     - Changes in height result in changes in weight.
     - but no mechanism
     - Gaussian variation result of growth history
**Scientific model**
- How does height influence weight?
  - W = f(H, U); U —> unobserved
  - *Weight is some function of height and unobserved stuff*
  - W = beta*H+U
4. Describing models
   - Conventional statistical model notation:
   - List the variables
   - Define each variable as a deterministic or distortional function of the other variables.
     - = -> deterministic, 
     - ~ -> distributed as
     - Equation for expected weight
     - U ->  Gaussian error with standard deviation sigma
     - H -> Uniform distribution 
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-21%2022.27.45.png)<!-- {"width":639} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-21%2022.29.14.png)<!-- {"width":639} -->
5. Statical models (Estimator)
   - Posterior distribution
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-21%2022.35.42.png)<!-- {"width":628} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-21%2022.39.21.png)<!-- {"width":628} -->
- *W is distributed normally with mean that is a linear function of H*
- **Grid approximate posterior**
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-21%2022.42.44.png)<!-- {"width":628} -->
- **Updating the posterior**
  - Hill —> More samples —> Small range for parameters
- **Enough grid approximation**
  - Quadratic approximation
- **Prior predictive distribution**
  - Priors should express scientific knowledges, but *softy*
  - Understand the implications of priors through simulation
  - What do the observable variables look like with these priors?
- **Sermon on priors**
  - There are no correct priors, only scientifically justifiable priors.
  - Justify with information outside the data *like rest of model*
  - Priors not so important in simple models.
  - *Very important/useful* in complex models
  - Need to practice now: simulate, understand
6. Validate model
   - **Simulation-based validation**
     - Bare minimum
       - Test statical model with simulated observation from scientific model
     - Golem might be broken
     - Even working golems might not deliver what you hoped.
     - Strong test: simulation-Based Calibration
   - Vary slope and make sure posterior mean tracks it
   - Use a large sample to see that it converges to data generating value
   - Same for other unknowns (Parameters)
7. Analyze data
   - Pairs plots
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-21%2023.16.20.png)
**Obey the law**
- First Law of statistical interpretation;
  - The *parameters are not independent* of one another and cannot always be independently interpreted.
- Instead
  - Push out *posterior predictions* and describe/interpret those.
**Flexible Linear Thermometers**
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-21%2023.27.05.png)
---
## L-04 Categories and Curves
### Drawing Inferences
- How to use statistical models to get at scientific estimates?
- Need to incorporate causal thinking into
  - How we draw the statistical models
    - Generative model + multiple estimates —> multiple estimators
  - How we process the results
    - Only very simple causal estimates show up in summary tables
    - Post-processing
- Linear regression can approximate anything
### Categories and Curves
- Linear models can do extra-linear things
- Categories: indicator & index variables
- Splines and other additive structures
- We require these tools to build causal estimators
### Categories
- How to cope with causes that are not continuous?
- Categories: discrete, unordered types
- Want to stratify by category:
  - Fit a separate line for each
- Think scientifically first
  - How A, B, C causally related?
- The causes are not in the data.
  - Knowledge 
  - Unobserved casuses
  - Ignorable unless shared
- Different causal questions need different statistical models
  - Causal effect of A on B?
  - Direct causal effect of A on B?
### From estimated to estimate
- Stratify by S
  - different estimate for each value S can take
### Drawing Categorical Owls
- Several ways to code categorical variables
  - indicator (0/1) variables
  - index variables: 1, 2, 3, 4…
- We will use index variables
  - Extend to many categories with no change in code
  - Better for specifying priors
  - Extend effortlessly to mult-level models
  - Look up —> Same priors distribution —> List
- Posterior means and predictions
- Always be contrasting
  - Need to *compute contrast*,  the difference between the categories
  - It is *never legitimate* to compare *overlap* in distributions
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2021.51.02.png)<!-- {"width":618} -->
- Must compute contrast distribution
- Weight contrast
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2021.54.52.png)<!-- {"width":618} -->
- Now we need to “block” association through H
- This means stratify by H.
  - indirect, direct
- Centering variables
  - Centering makes it easier to define scientific priors for alpha.
  - Easy to think about priors.
- Analyze the sample
- Compare the difference
  - Compute posterior for A and B
  - Plot subtract B from A
  - Plot distribution 
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2022.09.05.png)<!-- {"width":595} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2022.10.12.png)<!-- {"width":595} -->
- Summary
  - Easy to use with index coding
  - Use samples to compute relevant contrasts
  - Always summarize (mean. Interval) as the last step
  - Want **mean difference** and not **difference of means**.
### Curves from lines
- Linear models can easily fit curves
  - polynomials —> awful 
  - splines and generalized additive models —> less awful
- But this is not mechanistic
### Polynomial linear models
- Polynomial functions
- Problems
  - Strange symmetries,
  - Major: explosive uncertainty at edges
  - No local smoothing, only global
- Do not use
### Thinking vs. Fitting
- Linear models can fit anything (Geocentrism)
- Better to think
### Splines
- Basis-splines:
  - Wiggly function built from many *local functions*
- Basis function:
  - A local function
- Local functions make splines better than polynomials, but equally geocentric
### Going Local —> B-splines (Bayesian splines)
- B-Splines are just linear models, but with some weird synthetic variables
- Weights w are like slopes
  - ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2022.43.23.png)<!-- {"width":454.99999999999983} -->
- Basic functions B are synthetic variables.
- B values turn on weights in different regions
  ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2022.45.03.png)<!-- {"width":618} -->
- Obviously not linear
- Want function approximately right
- Fit spline as example
- But biological model would do a lot better
### Curves and Splines
- Can build very non-linear functions from linear pieces
- Splines are powerful geocentric devices
- Adding scientific information helps
  - e.g. Average weight only increases with height
  - e.g. Height increase with age, then levels off (or declines)
- Ideally statistical model has some form as scientific model
- Gaussian process
### Full Luxury Bayes
- We used two models for two estimates
- But alternative and equivalent approach is to use one model of entire causal system
- Then use joint posterior to compute each estimated
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2023.20.29.png)<!-- {"width":654} -->
- Causal effect is consequence of intervention
- Now simulate each intervention
- Simulating Interventions
  - Casual effect
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2023.23.04.png)
### Interface with linear models
- With more than 2 variables, scientific (causal) model and statistical model not always same
  - State each estimated
  - Design **unique statistical model** for each
  - **Compute** each estimated
  - One **stat model** for each estimated
- Or
  - State each estimated
  - Compute joint posterior for causal system
  - **Simulate** each estimated as an **Intervention**
  - One **simulation** for each estimated
---
## L-05 Elemental Confounds
### Correlation is commonplace
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2023.35.16.png)
### Association and Causation
- Statistical recipes must defend against confounding
- Confounds:
  - Features of the sample 
  - How we use it that mislead us
- Confounds are diverse
### Ye Olde Causal Alchemy
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2023.40.46.png)
### The fork
- Common cause
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2023.41.25.png)<!-- {"width":516} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2023.52.02.png)
- Estimated
  - Causal effect of marriage rate on divorce rate
- Scientific model
  - Fork: M <= A => D
  - To estimate causal effect of M, need to break the fork
  - Break the fork by stratifying by A
  - What does it mean to stratify by a continuous variable?
    - It depends
      - how does A influence D?
      - What is D = f(A, M)?
    - In a linear regression
      - D ~ Normal(mu, sigma)
      - mu = alpha + beta_1 * M + beta_2 * A
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-22%2023.57.14.png)<!-- {"width":510} -->
- Statistical model
  - Statistical Fork
  - We are going to *standardize* the data 
    - Mean is 0 std is 1
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%200.09.58.png)
- Standardizing 
  - Often convenient to standardize variables in linear regression
  - Standardize
    - Subtract mean and divide by standard deviation
  - Compute works better
  - Easy to choose sensible priors
- Analyze data
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%200.11.16.png)
- Slot is effect?
  - Never true for large models.
- Simulation interventions
  - A casual effect is a manipulation of the generative model, an intervention.
  - p(D|do(M)) means the distribution of D when we intervene (“do”) M
  - This implies deleting all arrows into M and simulating D
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%200.17.25.png)
- Causal effect of A?
  - How to estimate casual effect of A, P(D|do(A))?
  - No arrows to delete for intervention
  - Fit new model that ignore M, then simulate any intervention you like
  - Why does ignoring M work?
    - Because A -> M -> D is a “pipe”
## The Pipe
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%200.21.25.png)<!-- {"width":500} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2010.36.25.png)
- Post-treatment bias
  - Stratifying by (condition on) F induces post-treatment bias.
  - Mislead that treatment dose not work
  - **Consequences of treatment should not usually be included in estimator***
  - Doing experiments is no protection against bad casual inference.![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2010.44.40.png)<!-- {"width":435} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2010.46.52.png)<!-- {"width":500} -->
### Collider
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2010.48.12.png)<!-- {"width":510} -->
- Strong association after selection
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2010.54.50.png)
- Endogenous colliders
  - Collider bias can arise through statistical processing
  - Endogenous selection:
    - If you condition on (stratify by) a collider, creates phantom non-causal associations
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2011.07.48.png)
### The descendant
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2011.10.27.png)<!-- {"width":540} -->
- Descendants are everywhere
  - Many measurements are proxies of what we want to measure
  - Factor analysis
  - Measurement error
  - Social networks
- Unobserved confounds
  - Unmeasured causes (U) exist and can ruin your day
  - Estimand:
    - Direct effect of grandparents G on grandchildren C
  - Need to block pipe G -> P -> C
  - What happens when we condition on P?
---
## L-06 Good and Bad controls
- **Avoid being clever at all costs**
  - Being clever: unreliable, opaque
  - Given a causal model, can use logic to derive implications
  - Others can use same logic to verify and challenge your work
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2011.23.39.png)<!-- {"width":573} -->
- Causal thinking
  - In a experiment, we cut causes of the treatment
  - We randomize 
  - So how does casual inference without randomization ever work?
  - Is there a statistical procedure that mimics randomization?
    - P(Y|do(X)) = P(Y|?)
    - do(X) means intervene on X
  - Can analyze causal model to fin answer if it exists
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2011.35.47.png)
- Te causal effect of X on Y is not (in general) the coefficient relating X to Y.
- It is the distribution of  Y when we change X, averaged over the distributions of the control variables (here U)
### do-calculus
- For DAGs. Rules for finding P(Y|do(X)) known as do-calculus
- do-calculus says what is possible to say before picking functions
- Justifies graphical analysis
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2011.43.15.png)<!-- {"width":516} -->
- do-calculus is worse case
  - additional assumptions often allow stronger inference
- do-calculus is best case
  - if inference possible by do-calculus, does not depend on special assumptions
### Backdoor criterion
- A shortcut to applying some results of do-calculus
- Can be performed with your eyeballs
- Rule to find a set of variables to stratify by to yield P(Y|do(X))
  1. Identify all paths connecting the treatment (X) to the outcome (Y)
     - ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2011.52.26.png)<!-- {"width":403} -->
  2. Paths with arrows entering X are backdoor paths (non-causal paths, confounding paths) 
     - Right path
  3. Find adjustment set that closes/blocks all backdoor paths
     - Block the pipe: X not related to U | X 
     - Z knows all of the association between X, Y that is due to U
     - Coefficient on Z means nothing
  ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2011.59.18.png)<!-- {"width":582} -->
- Every part you added into the model is useless. 
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2012.02.03.png)<!-- {"width":582} -->
- All paths from X -> Y
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2012.02.28.png)<!-- {"width":582} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2012.06.34.png)<!-- {"width":582} -->
- www.dagitty.net
  - Ask computer how to prepare the adjustment  set
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2012.30.52.png)<!-- {"width":586} -->
### Backdoor Criterion
- Do-calc more than backdoors & adjustment sets
- Full Luxury Bayes:
  - **use all variables**, 
  - but in separate sub-models instead of single regression 
- do-calc less demanding;
  - find relevant variables
  - save us having to make some assumptions
  - not always a regression
### Good and Bad Controls
- Control variable
  - Variable introduced to an analysis so that a causal estimate is possible
- Common wrong heuristics for choosing control variables
  - Anything in the spreadsheet YOLO!
  - Any variables not highly collinear
  - Any pre-treatment measurement (baseline)
- A crash course in good and bas controls. Pearl 2021![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2012.50.23.png)<!-- {"width":463} -->
  1. List the paths
     - X -> Y: front door —> open
     - X <- u -> Z <- v -> Y: back door —> closed
     - Z is a bad control
     ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2012.53.35.png)<!-- {"width":403} -->
     - No back door, no need to control for Z
       - Change bzy = 0
     ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.00.06.png)<!-- {"width":618} -->
     - Significant variable here is not a guid to model structure.
     ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.04.36.png)
### **Do not touch the collider!**
- Bad control
- Colliders not always so obvious.
### Case-control bias (selection on outcome)![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.10.07.png)
### Precision parasite
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.15.58.png)
- No black doors
- But still not good to condition on Z
### Bias amplification
- X and Y confounded by u
- Something **truly awful** happens when we add Z
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.18.12.png)
- Why
  - Covariation X & Y requires variation in their causes
  - Within each level of Z, less variation in X
  - Confound u relatively more important within each Z
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.20.42.png)
### Good and Bad Controls Summary
- Control variable
  - Variable introduced to an analysis so that a causal estimate is possible
- Heuristics fail
  - Adding control variables can be worse than omitting
- Make assumptions explicit
### Table 2 Fallacy
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.45.44.png)<!-- {"width":463} -->
- The coefficients always depends upon causal assumptions.
  - Not all coefficients are causal effects
  - Statistical model designed to identify X -> Y will not also identify effects of control variables 
- **Table 2 is dangerous.**
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.50.59.png)<!-- {"width":454} -->
- Use backdoor criterion
  - X -> Y
  - S -> X, S-> Y
  - A -> X, X-> Y
  - A -> S -> Y, A -> X
- Stratified
  - A, S
- X
  ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.55.55.png)<!-- {"width":454} -->
- S
  ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.56.37.png)<!-- {"width":454} -->
- A
  ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2013.58.32.png)<!-- {"width":454} -->
- Not all coefficients created equal
- So do not present them as equal
- Options
  - Do not present control coefficients
  - Give explicit interpretation of each
- **No interpretation without causal representation**
---
## L-07 Fitting Over and Under
### Infinite causes, finite data
- Estimator might exist, but not be useful
- Struggle against causation:
  - How to use causal assumptions to design estimators, contrast alternative models
- Struggle data: How to make the estimators work
### Problems of prediction
- What function describes these points?
  - Fitting, compression
- What function explains these points?
  - Causal inference
- What would happen if we changed a points mass?
  - Intervention
- What is the next observation from the same process?
  - Prediction
### Leave one out cross validation
1. Drop one point
2. Fit line to remaining
3. Predict dropped point
4. Repeat 1 with next point
5. Score is error on dropped
### Bayesian Cross-Validation
- Wet use the entire posterior, not just a point prediction
- Cross-validation score is 
  - Log point wise predictive density (LPPD)
  - log —> Stable
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2022.53.53.png)
- Linear models are more general  than polynomial models
### Cross-validation
- For simple models, more parameters improves fit to sample
- But many reduce accuracy of predictions out of sample
- Most accurate model trades off flexibility with **overfitting**
### Regularization
- Overfitting depends upon the priors
- Skeptical priors have higher variance, reduce flexibility
- **Regularization**:
  - Function finds regular features of process
- **Good priors** are often tighter than you think.
  - Some priors are too constraining 
  - Sample size is too small
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2023.10.23.png)<!-- {"width":516} -->
### Regularizing priors
- How to choose with of prior?
- For *causal inference*, use science
- For *pure prediction*, can tune the prior using cross-validation
- Many tasks are a mix of inference and prediction
- No need to be perfect, just better
  - We need perfect model structure 
### Prediction penalty
- For N points, cross-validation requires fitting N models
- **What if you could estimate the penalty from a single model fit?**
- Good news
  - Importance sampling (PSIS)
  - Information criteria (WAIC)
- WAIC, PSIS, CV measure overfitting
- Regularization manages overfitting
- **None directly address causal inference**
- Important for understanding statistical inference
### Model Mis-selection
- **Do not use predictive criteria (WAIC, PSIS, SV) to chose a casual estimate**
  - Predictive criteria actually **prefer confound & colliders**
  - e.g: Plant growth experiment
- However, many analyses are mixes of inferential and predictive chores
- Still need help finding good functional descriptions while avoiding overfitting
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2023.25.26.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-23%2023.29.00.png)
### Outliers and Robust Regression
- Some points are more influential than others 
- Outliers
  - Observation in the tails of predictive distribution
- Outliers indicate predictions are possibly overconfident, unreliable
- The model dose not expect enough variation
- Dropping outliers is bad
  - Just ignores the problem
  - Predictions are still bad!
- It the model that’s wrong, not the data
  - First, qualify influence of each point
  - Second, use a mixture model (robust regression)
- Divorce rate example
### Mixing Gaussians
- Gaussian or student distribution
  - student-t regression more robust
### Robust Regressions
- Unobserved heterogeneity -> mixture of Gaussians
- Thick tail means model is less surprised by extreme values
- Usually impossible to estimate distribution of extreme values
- Student-t regression as default?
### Problems of Prediction
- What is the next observation from the same process? (prediction)
- Possible to make very good predictions without knowing causes
- Optimizing prediction does not reliably reveal causes
- Powerful tools (PISS, regularization)  for measuring and managing accuracy
---
## L-08 Markov chain Monte Carlo (MCMC)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2013.25.31.png)<!-- {"width":639} -->
### Real, latent modeling problems
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2013.28.57.png)<!-- {"width":292} -->
### Social networks —> Behavior relationship
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2013.31.39.png)<!-- {"width":392} -->
### Problems and solutions
- Real research problems:
  - Many unknowns
  - Nested relationships
  - Bounded outcomes
- **Difficult calculations**
### Computing the posterior
1. Analytical approaches: Often impossible
2. Grid approximation: Very intensive, more calculation
3. Quadratic approximation: Limited. Laplace approximation, similar to maximum likelihood
4. Markov chain Monte Carlo: Intensive
### MCMC
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2013.46.25.png)<!-- {"width":573} -->
- Usual use
  - Draw samples from a **posterior distribution**
- Istands
  - Parameter values
- Population size
  - Posterior probability
- Population size
  - Posterior probability
- Visit each parameter value in proportion to its posterior probability
- Any number of dimensions (Parameters)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2013.53.27.png)<!-- {"width":586} -->
- Chain
  - Sequence of draws from distribution
- Markov chain
  - History does not matter, just where you are now
- Monte Carlo
  - Random simulation
### Metroplis algorithm
- Simple version of MCMC
- Easy to write, very general
- often inefficient
### MCMC is diverse
- Metropolis has yielded to newer, more efficient algorithms
- Many innovations in the last decades
- Best methods use **gradients**
- We will use **Hamiltonian Monte Carlo**
### Basic Rosenbluth (aka Metropolis) algorithm
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.03.48.png)<!-- {"width":654} -->
### Gradients
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.10.20.png)<!-- {"width":659} -->
### Hamiltonian Monte Carlo
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.14.09.png)<!-- {"width":350} -->
- P 276 ~ 278
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.20.17.png)
### Calculus is superpower
- Hamiltonian Monte Carlo needs gradients
- How does it get them?
  - Write them your self
  - or …
- Auto-diff: Automatic differentitaion
- Symbolic derivatives of your model code
- Used in many machine learning approaches
  - Back-propagation is special case
- Stan
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.24.02.png)<!-- {"width":447} -->
### E.g.
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.30.51.png)<!-- {"width":586} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.32.37.png)<!-- {"width":586} -->
### Drawing the Markov Owl
- Complex machinery, but lots of diagnostics
1. Trace plots
   - Warmup and Samples
   - chains: more than 1 converged 
   - cores: computer core size
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.40.01.png)
2. Trace rank (Trank) plots
   - Rank orders of samples.
   - No china should tend to be below/ above others
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.45.20.png)<!-- {"width":474} -->
- NG
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.46.35.png)<!-- {"width":474} -->
3. R-hat convergence measure
   - When chains converge:
     - Start and end of each chain explores same region
     - Independent chains explore same region
   - R-hat is ratio of variances:
     - As total variance shrinks to average variance with chains, R-hat approaches 1
   - No guarantees: not a test
   - R-hat is 1.1 or 1.2 —> longer chain
4. Number of effective samples (n_eff)
   - Estimate of number of **effective samples**
   - How long would the chain be, if each sample was independent of that one before it?
   - When samples are **autocorrelated**, you have fewer **effective** samples?
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.52.41.png)<!-- {"width":463} -->
5. Divergent transitions
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.57.39.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2014.58.46.png)
- Harshness : how good the wine needs to be for this judge to rate it as average.
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2015.02.35.png)<!-- {"width":726} -->
- **The folk theorem of statistical computing**
  - When you have computational problems, **often there’s a problem with your model.**
### Divergent transitions
- Divergent transition:
  - A kind of rejected proposal
- Simulation diverges from true path
- Many DTs:
  - poor exploration & possible bias
- Will discuss again in later lecture
### El Pueblo Unido
- Desktop MCMC has been revolution in scientific computing
- Custom scientific modeling
- High dimension
- Propagate measurement error
---
## L-09 Modeling Events
### Flow forward
- Everything comes all at once
  - Scientific modeling, 
  - research design, 
  - statistical modeling, 
  - coding, 
  - interpretation, 
  - communication, 
  - the price of eggs
- Do not have to understand it all at once.
- Nobody ever does
### Modeling events
- Events:
  - Discrete, unordered outcomes
- Observations are counts
- Unknowns are probabilities, odds
- Everything interacts always everywhere
- A beast known as “log-odds”
### Admissions: Drawing the Owl
1. Estimates
2. Scientific models
3. Statistical models
4. Analyze
### Context and discrimination
- Wage discrimination
- Which path is discrimination?
  - Direct discrimination: Status-based or taste-based discrimination.
    - Requires strong assumptions
  - Indirect discrimination: structural discrimination
    - Requires strong assumptions
  - Total discrimination: experienced discrimination
    - Requires mild assumptions
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2022.44.01.png)
- Confounds!
  - Will ignore for now, but confounds will return
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2022.44.52.png)<!-- {"width":447} -->
### Generative model
- How can choice of department create structural discrimination?
- When departments vary in baseline admission rates?
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2022.52.42.png)
- Could do a lot better
- Admission rate depends upon size of application pool,, distribution of qualifications
- Should sample application pool and then sort to select admissions
- Rates are conditional on structure of applicant pool
### Modeling events
- We observe: Count of events
- We estimate:
  - Probability (or log-odds) of events
- Like the globe tossing model, 
  - but need “proportion of water”  stratified by other variables
### Generalized Linear Models
- Linear Models: 
  - Expected value is additive (“linear”) combination of parameters.
  - Y ~ Normal(mu, sigma)
  - mu = f(a, b, c)
- Generalized Linear Models:
  - Expected value is **some function** of an additive combination of parameters
  - Y ~ Bernoulli(p)
  - p = f(a, b, c) 
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.03.55.png)<!-- {"width":628} -->
### Links and inverse links
- f is link function
  - Links parameters of distribution to linear model
  - ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.06.16.png)<!-- {"width":250} -->
- f^(-1) is the **inverse** link function
  - ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.05.24.png)<!-- {"width":250} -->
### Distributions and link functions
- Distributions:
  - Relative number of ways to observe data, given assumptions about rates, probabilities, slopes, etc.
- Distributions are matched to constraints on observed variables
- **Link functions are matched to distributions**
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.14.26.png)
- Distributional assumptions are assumptions about constraints on obervations
- You cannot **test if you data are normal**
- P 312
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.17.55.png)
### Logit link
- Bernoulli/Binomial models most often use logic link
  - ![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.19.04.png)<!-- {"width":151} -->
  - log odds
  - odds = p / (1-p)
### From link to inverse link
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.23.01.png)<!-- {"width":447} -->
### Logit link ia a harsh transform
- log-odds scale
  - The value of the linear model.
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.24.46.png)<!-- {"width":240} -->
### Logistic priors
- logit(p) = alpha
- The logic link compresses parameter distributions
  - Anything above +4 = almost always
  - Anything below -4 = almost never
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.29.36.png)<!-- {"width":545} -->
### Estimated: Total effect of G
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.35.44.png)<!-- {"width":551} -->
### Estimated: Direct effect of G
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.38.03.png)<!-- {"width":551} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.40.33.png)<!-- {"width":700} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.43.54.png)<!-- {"width":700} -->
### Logistic and Binomial Regression
- Logistic regression:
  - Binary [0, 1] out come and logit link
  - Looooooong
- Binomial regression
  - Count [0, N] outcome and logit link
  - Aggregated
- **Completely equivalent for inference**
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-24%2023.52.08.png)<!-- {"width":350} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%200.00.23.png)
- What is the **average direct effect** of gender across departments?
  - Depends upon distribution of applications, probability woman/man applies to each department
  - What is the intervention actually?
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%200.04.53.png)<!-- {"width":510} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%200.06.10.png)
### Post-stratification
- Description, prediction and causal inference often require post-stratification
- Post-stratification
  - Re-weighting estimates for target population
- At a different university, distribution of applications will differ -> 
  - predicted consequence of intervention different
### Admission so for
- Evidence for discrimination? 
  - Yes
- Big structural effects, but
  1. Distribution of applications can be a consequence of discrimination 
     - Data do not speak to this
  2. Confounds likely
### Survival Analysis
- Counts often modeled as time-to-event
- Tricky, because cannot ignore censored cases
- Left-censored:
  - Do not know when time started
- Right-censored:
  - Observation ended before event
- Ignore-censored cases leads to inferential error
- Imagine estimating time-to-PhD:
  - Time in program before dropping out is info about rate
### Un-censored observation
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%200.47.51.png)
### Censored cats
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%200.49.27.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%200.52.02.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%201.00.31.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%201.02.53.png)<!-- {"width":407} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%201.03.09.png)
---
## L-10 Counts and Hidden Confounds
- 2024-02-25
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2012.22.06.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2012.23.03.png)
### Confounded Admissions
- Data simulation
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2012.31.56.png)
- Models
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2012.34.15.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2012.39.03.png)
### You guessed it: Collider bias
- Stratifying by D opens non-casual path through u
  - Can estimate total causal effect of G, but is not what we want
  - Can not estimate direct effect of D or G
- More intuitive explanation:
  - High ability G1s apply to discriminatory department anyway
  - G1s in that department are higher ability on average than G2s
  - High ability compensates for discrimination -> masks evidence
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2012.48.28.png)<!-- {"width":726} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.02.43.png)<!-- {"width":726} -->
### Citation networks
- Citation networks of members of NAS
- Women associated with lower lifetime citation rate
### Membership
- Elections to NAS
- Women associated with 3-15 times higher election rate, controlling for citations
- Restrict sample to NAS members, examine citations
  - If men less likely to be elected, then must have higher q, C to compare
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.05.02.png)<!-- {"width":435} -->
- Control for citations, examine elections to NAS
  - G is treatment.
  - C is a post-treatment variable.
  - If women less likely to be cited (bias),
    - then women more likely to be elected 
    - because they have higher q than indicated by C
  - Dangerous: Colliders for C
### No causes in; No causes out (Collider bias)
- Hyped parers with value estimates, unwise adjustment sets.
- Policy design through collider bias?
- We can do better
  - Strong assumptions required
  - Qualitative data useful
### Sensitivity analysis
- **What are the implications of what we do not know?**
- Assume confound exists, 
  - model its consequences for different strengths/kinds of influence
- How strong must the confound be change conclusions?
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.19.15.png)<!-- {"width":573} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.20.51.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.22.59.png)
- Show how big the assume confound would be.
### Sensitivity analysis
- **What are the implications of what we do not know?**
- **Somewhere between pure simulation and pure analysis**
- Vary confound strength over range and show how results change
  - -or- vary other effects and estimate confound strength
- Confounds persist — do not pretend
### More parameters than observations
### Oceanic Technology
- How is technological complexity related to population size?
- To social structure?
- Influence of **population size** and **contact** on **total tools**.
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.37.33.png)
## Modeling tools
- Tool count is not binomial: No maximum
- *Poisson distribution*:
  - very high maximum 
  - and very low probability of each success
- Here:
  - Many many possible technologies,
  - very few realized in any one place
### Poisson link is log
- Poisson distribution takes shape from expected value
- Must be positive
- Exponential scaling can be surprising
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.41.29.png)<!-- {"width":413} -->
### Poisson (poison) priors
- Exponential scaling can be surprising
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.43.58.png)<!-- {"width":540} -->
### Poisson priors
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.46.14.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.49.18.png)
- pPSIS is the **penalty**, the effective number of parameters
- How can a model with more parameters (m11.10) have fewer effective parameters
- m11.9 changes more when individual societies are dropped
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.51.46.png)
- log and natural scale
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.53.07.png)
## This model is wack
- Two immediate ways to improve the model
  1. Use a robust model: gamma-Poisson (neg-binomial)
  2. A principled scientific model
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2013.58.46.png)<!-- {"width":510} -->
- Different equation
  - Says how tools changes, not expected number
- What is the equilibrium?
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2014.01.12.png)<!-- {"width":350} -->
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2014.01.44.png)
- Still have to deal with location as confound
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2014.04.38.png)
### Count GLMs
- Distributions from constraints
- Maximum entropy priors
  - Binomial
  - Poisson
  - extensions
- More events types
  - Multinomial and categorical
- Robust regressions
  - Beta-binomial
  - gamma-Poisson
### Simpson’s Pandora’s Box
- Simpsons’s paradox:
  - Reversal of an association when groups are combined or separated
- No way to know which association (combined/ separated) is correct without causal assumptions 
- Infinite evils unleashed
### Berkeley Paradox
- Unconditional on department:
  - Women admitted at lower rate
- Conditional on department:
  - Women admitted slightly more
- Which is correct?
  - No way to know without assumptions
- Mediator (department)
- Collider + confounds (ability)
### Non-linear haunting
- In event models. Effect reversal can arise other ways
- Example: Base rate difference
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2014.23.25.png)![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2014.26.00.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2014.26.46.png)
![](Statistical%20Rethinking%202023%20%28Part-1%20L01~L10%29/%E6%88%AA%E5%B1%8F2024-02-25%2014.28.38.png)
- Does not know either way
- Does not accept the null
- No paradox, because almost anything can produce it
- People do not have intuitions about coefficient reversals
- Stop naming statistical paradoxes;
  - Start teaching scientific logic
