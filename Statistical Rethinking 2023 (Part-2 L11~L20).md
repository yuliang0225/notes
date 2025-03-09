# Statistical Rethinking 2023 (Part-2: L11~L20)
#study/statistical
## Refers
- ~[Statistical Rethinking 2023 Playlist](https://www.youtube.com/watch?v=FdnMWdICdRs&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus)~
- https://github.com/rmcelreath/stat_rethinking_2023?tab=readme-ov-file
- [[Statistical Rethinking 2023 (Part-1: L01~L10)]]
---
## L-11 Ordered Categories
### Trolley Problems
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.02.44.png)
- Estimand:
  - How do action, intention, contact influence response to a trolley story?
  - Treatment: X -> R (response)
  - How are influence of A?I/C associated with other variables?
  - ![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.06.35.png)<!-- {"width":404} -->
- Categories: Discrete types
  - car, dog, chicken
- Ordered categories: Discrete types with ordered relationships
  - bag good, excellent
- Objective distribution
  - Junk -> Okay -> Pretty good -> Good -> Awesome
### Ordered = Cumulative
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.13.59.png)
- cut-points: Gap
  - ![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.16.45.png)<!-- {"width":420} -->
  - ![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.18.15.png)<!-- {"width":420} -->
### Where is the GLM?
- So far just estimating the histogram
- How to make it a function of variables?
  - Stratify cut points
  - Offset each outpoint by value of linear model Phi
  - ![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.20.04.png)<!-- {"width":268} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.27.42.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.29.12.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.30.57.png)
### What about the competing causes?
### Total effect of gender:
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.34.06.png)
### Hang on! This is a voluntary sample
- Sample selection
- ![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2018.36.30.png)<!-- {"width":255} -->
- Conditioning on P makes E, Y, G covary in sample
### Endogenous selection
- Sample is selected on a collider
- Induces misleading associations among variables
- Not possible here to estimate total effect of G, But can get direct effect
- **Need to stratify by E and Y and G.**
### Ordered monotonic predictors
- **Education** is an ordered category
- Unlikely that each level has same effect
- Want a promoter for each level
- But how to **enforce ordering**, so that each level has large (or smaller) effect than previous?
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2019.01.35.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2019.02.44.png)<!-- {"width":420} -->
### Ordered monotonic priors
- How do we set priors for the delta parameters?
  - Delta parameters form as simplex
  - Simplex: vector that sums to 1
- Dirichlet(a) -> Delta
  - a = [2, 2, 2, 2, 2] or [1, 2, 3, 4, 5]
  - A distribution for distributions
  - sum to 1 -> Simplex
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2019.14.39.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2019.16.42.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2019.18.20.png)
### Complex causal effects
- Causal effects (predicted consequences of intervention) require marginalization
- E.g:
  - Causal effect of E requires distribution of Y and G to average over
- Problem 1:
  - Should not marginalize over this sample — cursed P!
  - Post-stratify to new target
- Problem 2:
  - Should not set all Y to same E
- E.g:
  - Casual effect of Y requires effect of Y on E, which we cannot estimate (P again!)
- No matter how complex, still just a **generative simulation** using **posterior samples**
  - Need generative model to plan estimation
  - Need generative model to compute estimates
### Repeat observations
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2019.36.27.png)
### Data science Task
- A second chance to get causal inference right
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2019.39.29.png)
- Quality of data is more important than quantity of data.
### Hitting the Target
- Basic problem: Sample is not the target
- **Post-stratification & Transport**:
  - Transparent, principled methods for extrapolating from sample to population
- Post-stratification requires casual model of reasons sample differs from population
- **NO CAUSES IN; NO DESCRIPTION OUT.**
### Cartoon example
- Proportions of sample != Age group
- not i.i.d.
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2019.54.07.png)<!-- {"width":429} -->
### Selection nodes
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2020.02.04.png)<!-- {"width":435} -->
### Selection ubiquitous
- Many sources of data are already filtered by selection effects
- Crime & health statistics
- Employment & job performance
- Museum collections
- Right thing to do depends upon causes of selection
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2020.06.47.png)<!-- {"width":556} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2020.08.23.png)<!-- {"width":556} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2020.09.52.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-25%2020.10.45.png)
### Honest methods for Modest questions
- Data have their bias
### Simple 4-step plan for honest digital scholarship
1. What are we trying to describe?
2. What is the ideal data for doing so?
3. What data do we actually have?
4. What causes the difference between 2 and 3?
5. [Optional] Is there a way to use 3+4 to do 1?
---
## L12 Multilevel Models
### Models with memory
- Multilevel models are models within models
  1. Model observed groups/ individuals
  2. Model of population of groups/indiciduals
- The population model creates a kind of memory
### Two perspectives
1. Models with memory learn faster, better
2. Models with memory resist overfitting
### Regularization
- Another reason for multilevel models is that they adaptively regularize
- Completer pooling
  - Treat all clusters as identical -> undercutting
- No pooling
  - Treat all clusters as unrelated -> overfitting
- Partial pooling
  - Adaptive compromise
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-26%2021.25.44.png)
### Cross validation and grid search (pSIS)
- Sample and out of sample
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2021.38.36.png)
### Automatic regularization
- Would not it be nice if we could find a good sigma without running so many models?
- Maybe we could learn it from the data?
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2021.43.12.png)
- Prior distubuition
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2021.44.52.png)<!-- {"width":379} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2021.45.29.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2021.47.25.png)
- Multi-level got regularization for hyper parameters
  - None multi-level models need cross validation to tune the parameters (CV)
  - WAIC and PWAIC -> smaller 
- **Adding parameters can reduce overfitting**!!!!!
  - What matters is structure, not number. !!!!!
### Stratify mean by predators
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2021.55.19.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2021.55.55.png)<!-- {"width":390} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2021.57.32.png)
### Multilevel Tadpoles
- Model of unobserved population helps learn about observed unites
- Use data efficiently, reduce overfitting
- **Varying effects** (Random effects)
  - Unite-specific partially pooled estimates
### Varying Effects Superstitions
- ==Varying effect models are played by superstition==
  1. ~~Units must be sampled at random~~ 
  2. ~~Number of units must be large~~
  3. ~~Assumes Gaussian variation~~
- **Gaussian prior can learn non Gaussian variations.**
### Practical Difficulties
- Varying effects are a good default, but …
  1. How to use more than one cluster type at the same time? For example stories and participants
  2. How to sample efficiently (MCMC)?
  3. What about slopes? Confounds?
### Random confounds
- When unobserved group features influence individually-varying causes
- Dizzying terminology:
  - group-level confounding, endogeneity, correlated error, econometrics
- **Group-level variables have direct and indirect influences**
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2022.15.19.png)<!-- {"width":441} -->
- Estimand: Influence of X on Y
- Estimator?
  1. **Fixed** effects model -> Draw back
  2. **Multilevel** model 
  3. Mundlak Machines -> Considered confound multilevel model
### Fixed effects model
- Estimate a different average rate for each group, without pooling
- Inefficient, but soaks up group-level (fixed) confounding (G)
- Problem:
  - Cannot identify any group-level effects (Z)
  - We cannot include Z
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2022.30.16.png)
- Z is over estimated by naive model.
### Multilevel model
- Estimate a different average rate for each group, partial pooling
- Better estimates for G, worse estimate for X
- Bonus: Can identify Z
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2022.35.24.png)
### Mundlak machine
- Estimate a different average rate for each group, partial pooling
- Include group average X
- Better X, but improper respect for uncertainty in X-bar.
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2022.37.50.png)<!-- {"width":168} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2022.41.04.png)
### Full Luxury Bayes
- Aka Latent Mundlak Machine
- Just express the generative model
- Treat G as unknown and use X to estimate 
- Two simultaneous regressions
  1. Estimate X|do(G)
  2. Estimate Y|do(X)
- Respects uncertainty in G
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2022.43.58.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-27%2022.45.57.png)<!-- {"width":313} -->
### Random confounds
- Should you use fixed effects?
- Should you include average X?
- Use a generative model, model the confound
- Confounds also vary at individual level -> No single solution
---
## L-13 Multilevel Advances
### Drawing the Bayesian Owl
1. Theoretical estimand
2. Scientific (Causal) models
3. Use 1 and 2 to build statistical models
4. Simulate from 2 to validate 3 yields 1
5. Analyze real data
### Multilevel Adventures
- Return to the start:
  - Start again, reinforce foundation
- Skim and index
  - Do not they to learn the details
  - Just acquaint yourself with possibilities
- Pick and Choose
  - Engage only with topics that interest you 
- Bayesian Flow
  - Just enough to keep moving
- CLuters
  - Kinds of groups in the data
- Features
  - Aspects of the model (parameters) that vary by cluster
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-28%2020.23.21.png)<!-- {"width":441} -->
- Add clusters
  - More index variables, more population priors
- Add features
  - More parameters, more dimensions in each population prior
### Varying effects as confounds
- Varying effect strategy:
  - Unmeasured feature of clusters leave an important on the data that can be measured by
    - repeat observations of each cluster
    - partial pooling among clusters
- Predictive perspective
  - Important source of cluster-level variation, regularize
- Causal perspective:
  - Competing causes or unobserved confounds
### Varying effects as confounds
- Causal perspective:
  - Competing causes or actual confounds
- Advantage over “fixed effect”
  - approach: Can include other cluster-level (time invariant) causes
- Fixed effects
  - Varying effects with variance fixed at infinity, no pooling
- Do not panic:
  - Make a generative model and draw the owl
### Practical Difficulties
- Varying effects are a good default, but …
  1. How to use more than one cluster type at the same time?
  2. How to calculate predictions
  3. How to sample chains efficiently
  4. Group-level confounding
### Example
1. Causes of interest
2. Competing causes
3. Relationship among causes
4. Unfortunate relationships among causes
5. A series of unfortunate relationships among causes
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-28%2021.15.02.png)<!-- {"width":496} -->
- Varying districts
  - Estimand:
    - C in each district, partially pooled
  - Varying intercept on each district
  - Another chance to understand partial pooling
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-28%2021.18.39.png)<!-- {"width":759} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-28%2021.19.09.png)
- Particle pooling shrinks districts with low sampling towards mean
  - Better predictions
  - No inference yet
- What is the effect of urban living?
  - District features are potential group-level confounds
  - Total effect of U passes through K
  - Do not stratify by K!
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-28%2021.31.24.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-28%2021.31.45.png)
### More priors, more problems
- Priors inside priors: “centered”.
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-28%2021.37.41.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-02-28%2021.42.55.png)
---
## L-14 Correlated Features
### Multilevel Adventures
- Clusters:
  - Kinds of groups in the data (districts)
- Features:
  - Aspects of the model (parameters) that vary by cluster (rural, urban)
- There is useful information to transfer
### Adding correlated Features
- One prior distribution for each cluster
- One feature: One-dimensional distribution
- Two features: Two-D distribution
  - ![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2020.01.26.png)<!-- {"width":295} -->
- N features: N-dimensional distributions
  - ![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2020.01.13.png)<!-- {"width":224} -->
- Hard part: 
  - learning associations
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2020.11.58.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2020.12.24.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2020.14.09.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2020.41.40.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2020.44.55.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2020.54.03.png)
### Correlated features
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2021.12.52.png)
### Correlated carrying effects
- Priors that learn correlation structure:
  1. Partial pooling across features
  2. Learn correlations
- Varying effects can be correlated even if the prior dose not learn the correlations
- Ethical obligation to do our best
### We cannot get enough effect samples
- Although, we make a longer MCMC.
- Non-centered would be better.
### Inconvenient posteriors
- Inefficient MCMC can be caused by steep curvature
- Hamiltonian simulation has trouble exploring surface
- “Divergent transitions”
- Transforming priors can help
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2021.32.43.png)
### Divergent transitions
- Why? Same step size not optimal everywhere
- High curvature = simulation cannot follow surface
- What can we do?
  1. Use a smaller step size
     - Make exploration slow
  2. Re-parameterize
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2021.37.18.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2021.42.11.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2021.46.05.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-02%2021.51.41.png)
### Transformed priors
- Both centered can non-centered better in different contexts
- Centered
  - Lots of data in each cluster
  - data probability dominant
- Non-centered:
  - Many clusters
  - sparse evidence
  - prior dominant
---
## L-15 Social Network
### What motivates sharing?
- T_ab and T_ba are not observable
- Social network:
  - Pattern of directed exchange
- Social networks are abstractions are not data
- What is a principled approach?
### Resist adhockery
- Do not control for non-independence
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2013.31.15.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2013.36.38.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2013.42.55.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.14.57.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.19.36.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.21.33.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.26.15.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.26.42.png)
- Network is uncertain, so all network statistics are uncertain!
### Social networks do not exist
- Varying effects are placeholders
- Can model the network ties
  - using dyad features
- Can model the giving/receiving
  - using household features
- Relationships can cause other relationships
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.33.41.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.35.41.png)
### Additional structure: Triangles
- Relationships tend to come in triangles
- Triangle closure
- Block models
  - Ties more common within certain groups 
  - family, office, stammtisch
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.37.18.png)<!-- {"width":149} -->![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.37.46.png)<!-- {"width":448} -->
### Varying effects as technology
- Social networks try to express regularities of observations
- Inferred social network is regularized, a structured varying effect
- Analogous problems:
  - phylogeny, space, heritability, knowledge, personality
- What happens when the clusters are not discrete but continuous?
  - Age, distance, time ,similarity
  - Next lesson
### Constructed variables are bad
- Folk tradition of building outcome variables as back-alley form of “control”:
  - ratios. Differences, transformations
- Body mass index = mass/height^2
- rates/ratios: per capital, per unit time
- differences:
  - change scores, difference from reference
- **All of these are usually bad.**
- Why NG
  - Effects are not the same
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.50.45.png)<!-- {"width":409} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2014.51.49.png)<!-- {"width":409} -->
### Constructed variables Are Bad!
- **Arithmetic is not stratification**
- Assumes a fixed relationship, when you should estimate
- Ignores uncertainty, e.g. rates
- Similar:
  - Do not use model predictions (residuals) as data
- Do
  - Use causal logic, justify, test
### Adhockery
- Long tradition of adhockery:
  - ad hoc procedures, intuition as justification 
- “we expect a correlation”
- ad hoc procedures not justified by probability theory go wrong
- Simple rule:
  - Model what you measure.
---
## L-16 Gaussian Processes
### Oceanic Technology
- Number of tool types associated with population size
- Spatial covariation:
  - Islands close together share unobserved confounds and innovations
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2021.29.26.png)
- Step by step, test them and understanding.
- k -> Kernel (45 covaiances)
- ![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2021.34.46.png)<!-- {"width":268} -->
- ![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2021.34.32.png)<!-- {"width":603} -->
  - 45 parameters are too more -> We need some method to model them
### Gaussian Processes
- A Gaussian Process is
  - an infinite-dimensional generalization of multivariate normal distributions
- Instead of conventional covariance matrix,
  - use a **kernel function** that generalizes to **infinite dimensions/observations/predictions**.
- Instead of conventional covariance matrix,
  - use a kernel function
- The kernel
  - gives the covariance between any pair of points as a function of their distance.
- Distance can be difference, space, time, etc
- **Continuous, ordered categories**
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2021.49.18.png)
### Kernel functions
- Quadratic (L2)
  - Gaussian
- Ornstein-Uhlenbeck
  - Non-Gaussian
- Periodic
  - circular, such as time orientation
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2021.58.53.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2021.59.41.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2022.00.36.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2022.01.34.png)
- Analysis by data
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2022.02.38.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2022.05.20.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2022.06.49.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2022.09.44.png)
### Phylogenetic regression
- Life history traits
- Mass g, brain cc, group size
- Real data
  - Much missing data, 
  - measurement error, 
  - unobserved confounding
### Causal salad in evolutionary ecology
- Phylogenetic comparative methods dominated by causal salad
- Causal salad:
  - Tossing factors into regression and interpreting every coefficient as causal
- Controlling for phylogeny:
  - Required but mindless
- Regression + phylogeny still requires causal model
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2022.53.30.png)
- No interpretation without causal representation
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2022.55.48.png)<!-- {"width":339} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2022.59.34.png)<!-- {"width":537} -->
- Two conjoint problems
  - What is the history (phylogeny)?
    - Gotten much better with genomics 
    - Problems:
      - **huge uncertainty** in the best case,
      - process **not stationary**,
      - no one phylogeny correct for **all traits**
    - Cultural/linguistic phylogenies unconvincing, 
      - need new inference tools
    - **Basic truth: Phylogenies do not exist**
  - How to use it to model causes?
    - Suppose we have a phylogeny.
    - No universally correct approach
    - Default approach is Gaussian process regression
- We can always replace a linear regression with an equivalent notational model where the outcome is multivariate normal.
  - No correlations assumptions
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2023.10.26.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2023.12.03.png)
### From model to kernel
- Evolutionary model + tree structure = pattern of covariation at tips
- Covariance declines with phylogenetic distance
- Phylogenetic distance:
  - Branch length from one species to another
- Common simple models
  - Brownian motion
  - Ornstein-Uhlenbeck 
    - damped Brownian motion
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2023.18.44.png)<!-- {"width":409} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2023.19.51.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2023.22.17.png)
- Stratify by M and G
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2023.24.21.png)
- Learned something
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2023.27.15.png)
### Phylogenetic regression
- Lingering problems:
  - What about phylogenetic uncertainty?
  - Do not these traits influence one another reciprocally over time?
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-03%2023.33.01.png)
### Gaussian Processes
- Partial pooling for continuous categories
- Very general approximation engine
- Causal theory -> Covariance kernel
- Sensitivity to kernel priors -> choose wisely
### Gaussian Possibilities
- Automatic relevance determination (ARD)
  - Multiple distance dimensions inside the kernel
- Multi-output Gaussian processes:
  - Draw vectors from kernel
- Telemetry, navigation:
  - Real-time tracking and error correction 
  - Kalman filter
---
## L-17 Measurement & Misclassification
### The importance of not being clever
- Being clever is unreliable and opaque
- Better to follow the axioms
- Probability theory provides solutions to challenging problems, if only we will follow the rules
- Often nothing else to understand
### Ye Olde Causal Alchemy
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%207.37.44.png)<!-- {"width":456} -->
### Measurement Error
- Many variables are proxies of the causes of interest
- Common to ignore measurement
- Many ad hoc procedures
- Think casually, lean on probability theory
### Myth: Measurement error only reduces effect estimates, never increase them
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%207.43.16.png)
### Modeling Measurement
- data -> WaffleDivorce
- State estimates D, M, A measured with error, error varies by state
- Two problems
  - Imbalance in evidence
  - Potential confounding
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%207.46.54.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%207.51.13.png)<!-- {"width":456} -->
### Thinking Like a Graph
- Regressions (GLMMs) are special case machines
- Thinking like a regression: 
  - Which predictor variables do I use?
- Thinking like a graph: 
  - How do I model the network of causes?
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%207.57.22.png)<!-- {"width":522} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%207.59.31.png)<!-- {"width":522} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%208.00.36.png)<!-- {"width":720} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%208.58.50.png)<!-- {"width":507} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%209.04.16.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%209.05.08.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%209.07.59.png)
### Unpredictable errors
- Including error on M increases evidence for an effect of M on D
- Why?
  - Down-weighting of unreliable estimates
  - Errors can hurt or help, but only honest option is to attend to them
### Misclassification
- Estimand: Proportion of children fathered by extra-pair men
- Problem: 5% false-positive rate
- Misclassification: 
  - Categorical version of measurement error
- How do we include misclassification?
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%209.27.44.png)<!-- {"width":323} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%209.28.45.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%209.31.44.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%209.32.57.png)
- log(P)
### Floating Point Monsters
- Probability calculations tend to underflow (round to zero) and overflow (round to 1)
- Solution
  - Calculate on log scale
- Ancient weapons
  - log_sum_exp
  - log1m
  - log1m_exp
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%209.36.35.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%209.38.11.png)
### Logarithms make sense
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%209.44.53.png)<!-- {"width":513} -->
### Devil in the details
- If p is close to zero, log(1-p) could evaluate to zero
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%2010.31.02.png)
### Log1p
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%2010.31.48.png)
### Log_sum_exp
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%2010.36.07.png)<!-- {"width":513} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-09%2010.39.29.png)
- Often unnecessary, but when you need them, no better defense
### Measurement Horizons
- Plenty of related problems and solutions
- Rating and assessment:
  - Judges and tests are noisy
  - item response theory and factor analysis
- Hurdle models
  - Thresholds for detection
- Occupancy models
  - Not detecting something dose not mean it is not there.
## L-18 Missing Data
### Missing data, found
- Observed data is special case
  - We trick ourselves into believing there is no error
- Most data are missing most of the time
- Missing data
  - Some cases unobserved
- Not totally missing: We know
  - constraints
  - relationships to other variables
### Missing data is workflow
- What to do with missing data?
- Dropping cases with missing values sometime justifiable.
- Right thing to do depends upon casual assumptions
- Imputation often beneficial/necessary
### Scenarios
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%209.19.19.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%209.23.22.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%209.24.44.png)
### Summary
1. Dog eats random homework
   - Dropping incomplete cases okay, but loss of efficiency
2. Dog eats conditional on cause
   - Correctly condition on cause
3. Dog eats homework itself
   - Usually hopeless unless we can model the dog
   - Survival analysis
### Bayesian imputation
1. Dog eats random homework
2. Dog eats conditional on cause
- Both imply need to impute or marginalize over missing values
- Bayesian imputation
  - Compute posterior probability distribution of missing values
- Marginalizing unknowns
  - Averaging over distribution of missing values
- Causal model of all variables implies strategy for imputation
- Technical obstacles exist!
- Sometimes imputation is unnecessary
  - e.g. discrete parameters
- Sometimes imputation is easier
  - e.g. censored observations
- Refer: P-517
### Phylogenetic regression
- Much missing data, measurement error, unobserved condoning
### Imputing primates
- Key ides
  - Missing values already have probability distribution s
- Express causal model for each partially-observed variable
- Replace each missing value with a parameter, let model do the rest
- **Conceptually weird, technically awkward**
### Dags
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%209.43.47.png)<!-- {"width":448} -->
- Assumptions
  - Species close to humans better studied
  - Larger species easier to count
  - Solitary species less studied
- Whatever the assumptions, 
  - our goal is to use the causal model to infer provability distribution of each missing value
- Uncertainty in each missing value cascades through the entire model.
### Bayesian imputation P2
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%209.50.34.png)<!-- {"width":726} -->
- Try to measure phylogenetic signa
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%209.52.41.png)<!-- {"width":726} -->
### Draw the Missing Owl
Let’s take it slow
1. Ignore case with missing B values
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.00.33.png)<!-- {"width":541} -->
2. Impute G and M ignoring models for each
   - Go easy on yourself
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.02.06.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.03.11.png)![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.13.13.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.14.20.png)
3. Impute G using model
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.16.32.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.16.58.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.21.06.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.22.51.png)
4. Impute B, G, M using model
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.24.06.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2010.25.08.png)
### Imputing primates
- Key idea
  - Missing values already have probability distribution
- Think like a graph, not like a regression 
- Imputation without relationships among predictors risky
- Even if dose not change result, it is our duty
---
## L-19 Generalized Linear Madness
### Generalized linear habits
- GLMs and GLMMs
  - Flexible association description machines
- With external causal model, causal interpretation possible 
- But only a fraction of scientific phenomena expressible as GLM(M)s
- Even when GLM(M)s sufficient, starting with throes solves empirical problems
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.04.10.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.04.25.png)
### How to set these priors?
1. Choose measurement scales
   - Measurement scales are artifice.
   - If you can divide out all measurement units (kg, cm), often much easier.
2. Simulate
   - Bert than no priors
   - Prior predictive simulation
   - Growth is multiplicative, log-normal is natural choice
   - mu in log-normal is mean of log, not mean of observed
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.19.08.png)
3. Think
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.27.14.png)
### Insightful error
- Not bad for a cylinder
- Poor fit for children
- In scientific model, poor fit is informative -> p different for kids
- Bad epicycles harder to read
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.28.24.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.28.38.png)
### Geometric People
- Most of the relationship H -> W is just relationship between length and volume
- Changes in body shape explain poor fit for children?
- Problems provide insight when model is scientific instead of purely statistical
- There is no empiricism without theory
### Social conformity
- Do children copy the majority?
  - If so, how does this develop?
- Problem
  - Cannot see strategy, only choice
- Majority choice consistent with many strategies
  - Random color 1/3
  - Random demonstrator 3/4
  - Random demonstration 1/2
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.47.00.png)
### State-Base model
- Majority choice does not indicate majority preference
- Instead infer the unobserved strategy (state) of each child
- Strategy space
  - Majority
  - Minority
  - Maverick
  - Random color
  - Follow first
- Categorical distribution
  - Probability of choice j (parameter is a vector)
  - ![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.54.26.png)<!-- {"width":448} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.53.48.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.55.07.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2016.56.38.png)
### State-Based Models
- What we want
  - Latent states
- What we have
  - Emissions
- Typically lots of uncertainty, 
  - but being honest is only ethical choice
- Large family
  - Movement, learning, population dynamics,
  - international relations, family planning, …
### Population Dynamics
- Latent states can be time varying
- Example:
  - Ecological dynamics
  - numbers of different species over time
- Estimand
  - How do different species interact
  - how do interactions influence population dynamics
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2017.35.18.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2017.39.02.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2017.46.40.png)
### Prior simulation
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2017.48.54.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2017.49.27.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-10%2017.50.04.png)
### Population Dynamics
- Ecologies much more complex
- Other animals prey on hare
- Without causal model, little hope to understand interventions
- Same framework very successful in fisheries management
### Science before statistics
- Epicycles get you only so far
- Scientific models also flawed, but flaws are more productive
- Theory necessary for empiricism
- Be patient; mastery takes time; experts learn safe habits
---
## L-20 Horoscopes
### Stargazing
Fortune telling frameworks:
- From vague facts, vague advice
- Exaggerated importance
Applies to astrologers and statisticians
Valid vague advice exists, not sufficient
- Statistical procedures acquire meaning from scientific models
- Cannot offload subjective responsibility to an objective produce
- Many subjective responsibilities
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2014.24.37.png)<!-- {"width":441} -->
### A typical scientific laboratory
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2014.27.17.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2014.27.57.png)
### Planning
- Goal setting
  - What for? Estimands <- Estimator <- Estimate
- Theory building
  - Which assumptions?
  - To construct an estimator
  - Causal model
- Justified sampling plan
  - Which data?
- Justified analysis plan
  - Which golems?
- Documentation
  - How did it happen?
- Open software & data formats
### Theory building
Level of theory building
- Heuristic causal models (DAGs)
  - Treatment and outcome
  - Other causes
  - Other effects
  - Unobserved causes
- Structural causal models
- Dynamic models
- Agent-based models
### Pre-Registration
- Prep-registration:
  - Prior public documentation of research design and analysis plan
- Goal:
  - Make transparent which decisions are sample-dependent
- Dose little to improve data analysis
- Lots of pre-registered causal salad
### Working
- Control
- Incremental testing
- Documentation
- Review
- Entire history open (Basic 4 steps)
  - Express theory as probabilistic program
  - Prove planned analysis could work (conditionally)
  - Test pipeline on synthetic data
  - Run pipeline on empirical data
### Professional Norms
- Dangerous lack of professional norms in scientific computing
- Often impossible to figure out what was done
- Often impossible to know if code works as intended
- Like pipetting by month
### Research engineering
- Control
  - Versioning, back-up, accountability
- Incremental testing:
  - Piece by piece
- Documentation
  - comment everything
- Review
  - 4 eyes on code and materials
### Versioning and testing
- Version control (git)
  - Database of  changes to project files, managed history
- Testing
  - Incremental milestones, test each before moving to next
- Most researchers do not need all gits’ features
- But do:
  - Commit changes after each milestone maintain test code in project
- Do not:
  - Replace raw data with processed data
More on testing
- Complex analyses must be built in steps
- Test each step
- Social networks (#15) as example
- Milestones:
  - synthetic data simulation
  - dyadic reciprocity model
  - add generalized giving/receiving
  - add wealth, association index
- e.g: rmcelreath /CES_rater_2021
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2020.29.46.png)
- datacarpentry.org/
- No excel, but csv
  - Careful primary data entry, okay with rules, tests
  - Never process data in excel, use code
  - Stop using excel
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2020.34.59.png)
### Reporting
- Sharing materials
- Describing methods
- Describing data
- Describing results
- Making decisions
### Sharing materials
- The paper is an advertisement; the data and its analysis are the product
- Make code and data available through a link, not “by request”
- Some data not shareable, code always shareable
- Archived code & data will be required
### Describing methods
Minimal information
1. Math-stats notation of stat model
2. Explanation of how #1 provides estimand
3. Algorithm used to produce estimate
4. Diagnostics, code tests
5. Cite software packages
- e.g.
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2020.46.18.png)
### Justify priors
- Priors were chosen through prior predictive simulation so that pre-data predictions span the range of scientifically plausible outcomes.
- In the results, we explicitly compare the posterior distribution to the prior, so that the impact of the sample is obvious.
### Justifying methods
- Naive reviewers
  - Good science dose not need complex stats
- Causal model often requires complexity
- Big data -> unit heterogeneity
- Ethical responsibility to do our best
- **Change discussion from statistics to causal models**
- Write for the editor, not the reviewer
- Find other papers in discipline/journal that have Bayesian methods other than similar models (Bayesian or not)
- Explain results in Bayesian terms, show densities, cite disciplinary guides
- **Bayes is ancient, normative, often the only practical way to estimate complex models.**
### Describing data
Effective sample size function of estimand and hierarchical structure
Variables measured at which levels?
- Missing  values!
### Describing results
- Estimands, marginal causal effects
- Warn against causal interpretation of control variables
- Densities better than intervals
  - Sample realizations often better than densities
- Figuress assist comparisons
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2021.06.04.png)
- https://knightcenter.utexas.edu/JC/courses/DATA0819/HowChartsLie_INTRODUCTION.pdf
### Making decisions
- Academic research:
  - Communicate uncertainty, conditional on sample and models
- Industry research:
  - What should we do, given the uncertainty, conditional on sample and models?
- Also
  - Does my boss have any idea what uncertainty means, or does he think that’s the refuge of cowards?
- Bayesian decision theory
  - State costs and benefits of outcomes
  - Compute posterior benefits of hypothetical policy choices
- Simple example in  Chapter 3
- Can be integrated with dynamic optimization
### Science reform
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2021.14.58.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2021.16.36.png)
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2021.19.25.png)<!-- {"width":378} -->![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2021.21.33.png)<!-- {"width":365} -->
![](Statistical%20Rethinking%202023%20%28Part-2%20L11~L20%29/%E6%88%AA%E5%B1%8F2024-03-18%2021.22.04.png)<!-- {"width":355} -->
### Horoscopes for Research
- No one knows how research works
- But many easy fixes at hand
  - No stats without associated causal model
  - Prove that your code works (in principle)
  - Share as much as possible
  - Beware proxies of research quality
- **Many things you dislike about academia were once well-intentioned reforms**
---