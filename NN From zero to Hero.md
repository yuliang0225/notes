# NN: From zero to Hero
**[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)**
## 11-Deep dive into LLMs like ChatGPT
- Ageneral audience deep dive into the Large Language Model (LLM) AI technology
### Refers
- https://www.youtube.com/watch?v=7xTGNNLPyMI
### Pretraining model —> base model (3 month)
- Try to get Ton of text from the internet form publicly avaliable sources.
  - Large high quality + diversity documents
  - 15T tokens, 44 TB
- CommonCrawl
  - small data —> start point
- [FineWeb: decanting the web for the finest text data at scale - a Hugging Face Space by HuggingFaceFW](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- Step 1 download text form web
  - ![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-18%2010.15.46.png)<!-- {"width":700} -->
  - URL filters —> blacklist
  - Text extraction —> raw html —> text
  - Language filtering —> what language the page say —> small data for some language
  - Dedup —> by using minhash
- Step2 tokenization —> pre train by other dataset
  - NN need
    - one-dimensional sequence of symbols (tokens)
    - finite set of symbols (tokens)
    - bits trade off the symbols (more symbols)
    - group of some bytes —> a symbol —> sequence of bytes —> shrink the length of sequence —> tokenization (pair of symbols with hight frequency)
  - From text to symbols (tokens) —> tokenization —> chunck of text
- Step 3 NN traning
  - maximum tokens —> crop them
  - predict the next token
  - init parameters —> loss —> update parameters
  - [LLM Visualization](https://bbycroft.net/llm) —> understand the information flows
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-18%2010.43.57.png)<!-- {"width":662} -->
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-18%2010.45.26.png)
- Step 3 NN inference
  - To generate data, just predict one token at a time.
  - predicting —> feeding back —> predict 
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-18%2010.57.02.png)<!-- {"width":580} -->
- GPT-2: training and inference
  - context length is limited
  - https://github.com/karpathy/llm.c/discussions/677
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-18%2012.02.07.png)
- LLama 3: base model inferece (405 B parameters + 15T tokens)
  - model.py file —> network, transformers
  - parameters value —> just numbers
  - Hyperbolic —> serving base model
  - base model
    - not an assistant yet, just a token autocomplete.
    - a stochastic system
    - knowledge is vageue and probalistic and statistical.
    - text —> copy
  - few shot prompt —> never seen text —> prompting all language model
  - Prompt engineer
    - human assistant —> conversation pairs
    - One way to create an assistant 
### Post-traning (3 hours) —> Fine tuning 
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Cheapter and still extremely important stage
  - From a base model to an assistant.
- Conversations —> data set —> human labels (Ideal response) —> small data set
  - multi-turns
  - how in a system should interact with the human
  - we want to program the assistant and its behavior in these conversations
  - **we cannot programming these explicity in code**
  - Implicitly programming the assistant.
- Data strcutures —> conversations (protocols)
  - encode and decoded from tokens.
  - spectial token: [Tiktokenizer](https://tiktokenizer.vercel.app/?encoder=gpt2)
    - <|im_start|>user<|im_sep|>dummy<|im_end|>
    - These tokens are our model need to learn
  - Conversation encoding —> tokens —> inference
    - Create a prompt and also give the ideal prompt
    - Expert labeling instructions —> human heavy process
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-18%2012.35.26.png)
- Conversation datasets
  - https://atlas.nomic.ai/
- Statistical simulation of a labeler that was hired by openAI
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-18%2013.51.00.png)
- Hallucinations
  - Model can not say I do not know.
  - Add some data into the model.
  - We can generate some questions by using LLM
  - We can search some key words. —> add the results in to the conversation.
    - Create some excamples to show model how to use these tools.
    - Anwser as a memory, sometime we need to tell our model just use the memory.
    - sources, citations
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-18%2013.56.57.png)<!-- {"width":546} -->
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-18%2014.14.20.png)
- System message —> inserted into to your conversations
  - Hardcode and remind the model: You are a model xxxxx and your name is xxxx
  - Documents of the model
- Models need tokens to think
  - Chain of thought
  - Teach a model to solve math probelems.
  - If we can get the answer of the model directly, we are try to basically guess the anwser by some tokens. —> Overfittigs, model without thought.
  - We should make the anser in the end —> model can use generated tokens to understand the questions.
- Add `use code` in the end of the prompts
  - Make the model think —> using pythong or other code —> more better
- Models cannot count
- Models are not good with spelling —> Token is not a word
  - Remeber they see tokens (text chunks), not indicidual letters!
- Millions of conversation from different topics.
  - Human prompts —> Human response
  - Help from some LLMs + revised by human curation
### Post-training: reinforcement learning
- SFT model —> An assistant, trained by supervised finetuning
- Take our LLM to school.
  - Train by using expert —> we can try to imitate the expert
  - practice probelems —> no solutions —> only have the final answer
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-19%2014.31.00.png)<!-- {"width":685} -->
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-19%2014.38.29.png)
- Try many types of solutions
  - thousand of solutions
  - some solutions that lead to correct answers
  - A model can decided which solutions is bertter or not
  - What kind of token sequences lead it to a better solution
  - Guess and check
- How to train it
  - stage by stage 
- Deepseek
  - [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-19%2017.43.55.png)
- Chians of thought
  - An mergent property of the optim of the optimization
  - The model is discovering ways to think —> Cognitive strategies
  - Thinking process —> trying different ways
  - reasoning traces
  - Difficult problems
- RF
  - Stornger than human
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-19%2017.45.46.png)<!-- {"width":432} -->
- Discovery knowledge
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-19%2017.53.30.png)<!-- {"width":700} -->
- RLHF (human upside): Human feedback 
  - which one is better —> rank
  - Reward model —> Score
  - discriminator — generator gap
  - in many cases, it is much easier to discriminate to generate.
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2013.33.22.png)<!-- {"width":712} -->
- RLHF downside
  - we are doing RL with respect to a lossy simulation of humans. It might be misleading!
  - Even more subtle;
    - RL discovers ways to game the model
  - Overfiting the reward model.
    - inf training is NG.
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2013.42.58.png)<!-- {"width":712} -->
#### PREVIEW OF THINGS TO COME
- multimodal (not just text but audio, images, video, natural conversations)
  - by adding some tokens
  - tokenized that, patches
- tasks -> agents (long, coherent, error-correcting contexts)
  - supervised them
- pervasive, invisible
- computer-using
  - control keyboard or mouse
- test-time training?, etc.
  - training stage —> In context learning —> chain of thinking
#### WHERE TO FIND THEM
- Proprietary models: on the respective websites of the LLM providers
- Open weights models (DeepSeek, Llama): an inference provider, e.g. TogetherAI
- Run them locally! LMStudio
  - [Discover, download, and run local LLMs](https://lmstudio.ai/)
- [Chatbot Arena \(formerly LMSYS\): Free AI Chat to Compare & Test Best AI Chatbots](https://lmarena.ai/)
  - try these models and see which one performs better
- https://buttondown.com/ainews/archive/
- Simulation of human label —> thinking models (using RL)
---
## Deepseek v3
- https://www.youtube.com/watch?v=M9TzN-0Raz0
- Mixture of Experts
  - 37 B * N = 671 B
  - Fast
- R1 —> RL —> Chain of thought  (basic model is llama)
- Distill 
  - Teacher —> Student model —> loss (median loss)
- Only Open inference code, no training code. Open model paramters.
- No cuda —> more fast. —> nsloss —> basic code.
  - [Compiled language](https://en.wikipedia.org/wiki/Compiled_language)
  - Training —> Adam
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.11.47.png)<!-- {"width":558} -->
- FAISS
- MOE
  - Questions —> Router —> Experts
  - To prevent dead experts
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.16.00.png)
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.19.00.png)
- Multi-token prediction
  - to predict next 1, 2, 3, 4 tokens
  - cutoff some low prob predictions
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.20.07.png)
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.22.31.png)
- Engineering optimized
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.23.04.png)
- Nvidia
  - CPU lead GPU —> PCIE —> pass data
  - Share DRAM
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.25.38.png)
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.42.03.png)
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.43.42.png)
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.44.19.png)
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.46.48.png)
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202025-02-20%2018.48.03.png)
- LLamacpp
  - https://github.com/ggml-org/llama.cpp
  - To prediction —> cpp 
- Matrix multiply QKV
  - http://youtube.com/watch?v=HqTaOzvUuuk
![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2025-02-20%20%E4%B8%8B%E5%8D%887.18.07.png)<!-- {"width":670} -->
![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2025-02-20%20%E4%B8%8B%E5%8D%887.21.53.png)<!-- {"width":662} -->
- Deepseek hardware
- PTX —> Fast
---
## 07-Let’s build GPT: from scratch, in code, spelled out.
#Study
### Refers
- https://www.youtube.com/watch?v=kCc8FmEb1nY&list=WL&index=25&t=11s
- https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing
- https://github.com/karpathy/ng-video-lecture
- [Andrej Karpathy](https://karpathy.ai/)
- https://www.youtube.com/watch?v=zjkBMFhNj_g
### Links:
- Google colab for the video: [https://colab.research.google.com/dri...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqazJzNFZOTnozVWZnTjdINHB4dF9WQ2t2RExNQXxBQ3Jtc0tucnQ2cTdoSXBKaGNMMDRqT0hkMGxvWmtKT2VNRkVtQlB3QUh3TUJJYmFEQVZDMU1PZ1Mwb0VNaHRDay1pY2daWlJpNHFRREVsTVg0b1BQX0I3VWxRdkl6U210dk5zUzE1TEFkNXdrcndPT0c4R2lFWQ&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-%3Fusp%3Dsharing&v=kCc8FmEb1nY)
- GitHub repo for the video: [https://github.com/karpathy/ng-video-...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbHNUTHVNN1QtZlV0VktWT2RrU0tUMTJ4LVRKZ3xBQ3Jtc0trS1dsTkpoN2FQamRTR2o2dWZGa3NhTXMzTDdOd1llU0ItZmhQdkdFU1dZWEo4STZwSDlhYmNLWERxUW1VZld4QlpDMUhrMERuRUhXTmxsWG1vNVNSTHJkZWZVanhkSW1fSHZOZHlVcDBFd09wMWpPdw&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fng-video-lecture&v=kCc8FmEb1nY)
- Playlist of the whole Zero to Hero series so far: [](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=0s)![](NN%20From%20zero%20to%20Hero/yt_favicon.png)[• The spelled-out intro to neural netwo...](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=0s)
－ - nanoGPT repo: [https://github.com/karpathy/nanoGPT](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbTB3MlJLTkd2elotazM2SU1Pb3JFaWRuOHdWQXxBQ3Jtc0tuMDNod0ZfYXQxeURjdVpLQXlWaVZLbUdibXA3VnhYeWlZXzNPV3o0ZUljQklvd3JLdG04TVZyeXNpcWN5YXVDWmMxamVVeFRuSGkxblRDelN4MFRISDFQRFE3RGhDajRRVE1kX3lvQ3YxeE9kSGhkcw&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2FnanoGPT&v=kCc8FmEb1nY)
### Supplementary links:
- Attention is All You Need paper: [https://arxiv.org/abs/1706.03762](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa1J0TnZENWpNSkp0VGZ5MFhDdXFtWE1kb3NvQXxBQ3Jtc0ttVUhJVkhhTTFTY1Q2aDB5a0hQNXRtaHo3YjhnUDI2MzhCcnNMQWhRalMxcGwyS00yS1kzdWlqNEJzVThLOWtXXzB2RzZubFZfZHhCMUNiOThnRUFpeDc5VEFUSTB6M3hqVTRhT1B0dTY4dG5UOXZ1OA&q=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762&v=kCc8FmEb1nY)
- OpenAI GPT-3 paper: [https://arxiv.org/abs/2005.14165](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqblh4QTJ2WjEzU1FhejdOYzhRemlYYm9jaG00Z3xBQ3Jtc0ttaXo5bDFBSDNxelhLVllMMVpwTGJFSy1kQkJ3SF9IWEhmUFFIX2U3aGVVdDlSVmcyRXA4b2c5OW0wVlJIeERkM3dFb0h5eUFvS05hQUNEa1dXRjJURHc0U3ZXSm95T0d3SW8xb2Z0OTgxT2IyYjFkQQ&q=https%3A%2F%2Farxiv.org%2Fabs%2F2005.14165&v=kCc8FmEb1nY) 
- OpenAI ChatGPT blog post: [https://openai.com/blog/chatgpt/](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa2NiVVF0d2MzcGxUMENEWFNhZnVsdk1nd3IwUXxBQ3Jtc0tuTlZDeHl6ZEhqNFZ6MTlMbllQRXpkaTRpaWgzQldQVE1XNGwzRFNUdUZqMV9oLXEwRGktT215MkdBQUFEanpJbU1jZFN1SmxfZUlxVm5ia0JFbWNROHBjTVJMYnBkMk1nd1QwQ2g1bHAtMDFic0g2bw&q=https%3A%2F%2Fopenai.com%2Fblog%2Fchatgpt%2F&v=kCc8FmEb1nY)
- The GPU I'm training the model on is from Lambda GPU Cloud, I think the best and easiest way to spin up an on-demand GPU instance in the cloud that you can ssh to: [https://lambdalabs.com](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa2U1X25rNkl3RGVKRFRNN04wWVpiak5DRi15Z3xBQ3Jtc0tudE1kdU9hZWpSYlRYcGY2RzNRbFc2TVc1TmtGMC1MS1J4eFE2OUlnN2tCNG5EV2pVc2JSbjJyQkNLeXVUZUlxY3k3ZGlncEtQUmxSRDVjb1Y2TFNpVTBOdkROTHEtaG5xTjQtT29NTmROWVpoYkFhcw&q=https%3A%2F%2Flambdalabs.com%2F&v=kCc8FmEb1nY) . If you prefer to work in notebooks, I think the easiest path today is Google Colab
### Data set
- Shakespeare
- Task: predict next works -> generation
- Token number
- nano GPT
### Word embeddings
- Convert string to vector
  - BPE: [Byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)
    - Byte Pair Encoding (BPE) 是一种数据压缩技术，最初用于文本压缩，后来在自然语言处理（NLP）中，特别是在机器翻译和文本生成任务中，被广泛用作一种子词分割算法。
    - BPE 在处理词汇多样性和罕见词问题方面非常有效，尤其是对于那些词汇量极大或包含大量罕见词的语言模型。
    - 在NLP中，BPE的基本思想是**将常见的字符或字符序列合并为一个单独的新符号**，这样可以有效减少整个数据集中的不同符号的总数。通过这种方式，BPE能够将复杂的词或词组分解为更简单的子单位，这些子单位可以共享，从而提高模型的泛化能力。
    - 在机器学习，特别是在NLP领域，BPE让模型能够有效处理大量的未知词汇。通过将未见过的单词分解为已知的子单元，模型可以更好地处理和理解这些单词的含义。这种方法已经成为许多现代NLP模型，如BERT和GPT系列模型，处理词汇的标准技术之一。
  - Google: sentencepiece
    - [google/sentencepiece: Unsupervised text tokenizer for Neural Network-based text generation.](https://github.com/google/sentencepiece)
    - sub_word unit level for embedding
  - OpenAI: tiktoken
    - [openai/tiktoken: tiktoken is a fast BPE tokeniser for use with OpenAI's models.](https://github.com/openai/tiktoken)
### Attention
```python
# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x) # (B,T,C)
out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

out.shape (B, T, head_size)
```
### Notes:
* Attention is a **communication mechanism**. 
  * Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
* **There is no notion of space.** 
  * Attention simply acts over a set of vectors. 
  * This is why we need to positionally encode tokens. (POE)
* Each example across batch dimension is of course processed completely independently and never "talk" to each other.
* In an "encoder" attention block just delete the single line that does masking with `tril`, allowing all tokens to communicate. —> Causal effects —> Auto regression 
  * This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
* `self-attention` just means that the keys and values are produced from the same source as queries. 
  * In `cross-attention`, the queries still get produced from x, 
  * but the **keys and values come from some other**, external source (e.g. an encoder module)
* "Scaled" attention additional divides `wei` by `1/sqrt(head_size)`.
  * This makes it so when input Q,K are **unit variance**, `wei` will be unit variance too and `Softmax` will stay diffuse and not saturate too much. Illustration below
  * Control vars
- Why multi-head attention?
  - **Different aspect of features**
  - Learn more information form data
  - More fast, multi smaller matrixes is better than a large matrix. Parallel computation
```python
k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5

# gets too peaky, converges to one-hot
```
### Batch norm
- For each batch data, each rural saw the Gaussian (mean=0, std=1) data
- Normalizing columns
- New parameters: Gamma and beta
### Drop out
- Before residual connection back
- regularization technique
- % intermediate calculations are disabled and dropped to zero
  - Enforce some nodes learn some weights, not zero.  
- Overfitting
### Hyper parameters
```python
# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3 # smaller to prevent gradient explosion
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4 # n_head * n_embd == word embedding dim
n_layer = 4 # Attention layers
dropout = 0.0
# ------------
```
### Training
- **#1 Pre-training step**
  - word generation
  - get pre-train parameters —> word/position embeddings and attention parameters
  - tokens sub-words chunks 300 billion tokens 
    - Today is more
  - GPT3 paper
    - 175 B 
- **#2 Collect comparison data and train a reward model**
  - binary model
  - sample set [Q, A1, A2] A1 is better than A2
  - use the model output logit as reward score model
- **#3 Optimize a policy against the reward model using the PPO reinforcement  learning algorithm**
  - Try to generate text make reward model with higher score
### Code check
```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # Merge for each head embeddings
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension
		# n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
```
- 在 PyTorch 中，`register_buffer` 是 `nn.Module` 的一个方法，用于将不作为模型参数（nn.Parameter）的张量注册到模型中，通常用于存储一些在训练或推理过程中需要固定或保持不变的状态信息（如均值、方差、掩码等）。
  - 这些张量不参与梯度计算和优化，但会跟随模型的设备移动（如 .cuda() 或 .cpu()）、保存和加载（如 model.save() 和 model.load()）操作。
- `register_buffer`的使用场景：
  1.**批归一化（BatchNorm）中的均值和方差**：
  * 在 Batch Normalization 中，均值和方差是在推理过程中使用的，但它们并不参与反向传播，也不是可学习的参数。因此，它们通常会通过 register_buffer 来注册到模型中。
  2.**掩码矩阵**：
  * 在一些神经网络（如 Transformer）中，常常需要使用掩码来屏蔽某些位置的数据。例如，在自注意力机制中，可以使用掩码来忽略某些位置的计算。这个掩码矩阵不会随梯度更新，但需要在模型中保持，并随着设备的移动而移动。
  3.**指数移动平均（Exponential Moving Average, EMA）**：
  * 一些模型会使用指数移动平均来跟踪某些统计量或模型参数的变化。虽然这些量会随着训练过程更新，但它们并不需要通过反向传播来学习。这种情况下，使用 register_buffer 来存储这些 EMA 值是非常合适的。
---
### PTM
![](NN%20From%20zero%20to%20Hero/image.png)
---
## Neural Networks: Zero to Hero
- [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html)
### 01-The spelled-out intro to neural networks and backpropagation: building micrograd
- micrograd on github: [https://github.com/karpathy/micrograd](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbFJRN1dsTWhvRE1RaUdrX3Jyc1ZzcktPX2pmZ3xBQ3Jtc0tuS0ttRUVfRU9FdHctVUdxbW0wa2RaUjlMOUJoZ3ZaSktYNVVqT2Z6X3Q5c3BGOFZHTHdrOTRUeTBrNlhrM0pSZXJfYTJsMWUxWl9TUi1WSmRRVEp6SW5UYXBmMEtHckVZdTlDNWZfUFpvMDlXanBiNA&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fmicrograd&v=VMj-3S1tku0)
- jupyter notebooks I built in this video: [https://github.com/karpathy/nn-zero-t...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVBpenhfSjNCQ3JLOHpMWUZiQ1dDRnlCTW1LQXxBQ3Jtc0tuUTBtQ0l0MTdJOUFsYTFfdHc5eWdqRzFmdEUwaDZ1N2NqcEJsakpwQmFnVVJGZTN4T0NYQTg5TnhWcUc4a2pxS3duZDlsWXo0RWRIdmt5bEw0M1lOSzRYN2dHdzlZQ1djaU9sTkU4SlNBaWxQZlVJSQ&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fnn-zero-to-hero%2Ftree%2Fmaster%2Flectures%2Fmicrograd&v=VMj-3S1tku0)
- [機械学習のための数学](https://hinaser.github.io/Machine-Learning/math-for-ml.html)
- Mistakes
  - Do not try to overfit a single batch first.
  - Forgot to toggle train/eval mode for the net
  - Forgot to .zero_grad() before .backward()
  - Passed softmaxed outputs to a loss that expects raw logits
- Define new autograd functions
  - lego block
  - try new types of functions
```python
# optimization
for k in range(100):
    
    # forward
    total_loss, acc = loss()
    
    # backward
    model.zero_grad() # Care about here
    total_loss.backward()
    
    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

```
### 02-The spelled-out intro to language modeling: building makemore
- https://github.com/karpathy/makemore
- Borad casting
```python
# 27 * 27
#  1 * 27 --> Wrong results!!!!!
# Why the keepdim is necessary!
# Normazing the columns or the rows
# Respect for boradcasting, be careful check

# P = N.float()
# P = P / P.sum(1)

# Check the sum is 1 or not
# P[0].sum() != 1
# P[:, 0].sum() == 1
```
- softmax
![](NN%20From%20zero%20to%20Hero/image%202.png)<!-- {"width":231} -->![](NN%20From%20zero%20to%20Hero/image%203.png)<!-- {"width":449} -->
- exp() -> to make val > 0
- normalized by sum -> to make sum as 1
- convert score to probability
- Loss function
  - Find the parameters with the lowest loss
  - Goal: maximize likelihood of the data w.r.t. model parameters (statistical model)
  * equivalent to maximizing the log likelihood -> because log is monotonic
  * equivalent to minimizing the negative likelihood
  * equivalent to minimizing the average negative log likelihood
  * log(a*b*c) = log(a) + log(b) + log(c)

```python
# evaluation the bigram model
# MLE
# log(p) likelihood
# log(a*b*c) = log(a) + log(b) + log(c)
# nll = -sum(log(p)) -> loss fucntion -> lower is better

log_likelihood = 0.0
n = 0
for w in words[:3]:
    chs = ['.'] + list(w) + ['.']
    for c1, c2 in zip(chs, chs[1:]):
        ix1 = stoi[c1]
        ix2 = stoi[c2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f"{c1}{c2}: {prob:4f} {logprob:4f}")

nll = -log_likelihood/n
print(f"{log_likelihood=:4f}")
print(f"{nll=:4f}")
```

- Loss
  - Loss was the single number that summarized the quality of the neural net.
```python
nlls = torch.zeros(5)
for i in range(5):
  # i-th bigram:
  x = xs[i].item() # input character index
  y = ys[i].item() # label character index
  print('--------')
  print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
  print('input to the neural net:', x)
  print('output probabilities from the neural net:', probs[i])
  print('label (actual next character):', y)
  p = probs[i, y]
  print('probability assigned by the net to the the correct character:', p.item())
  logp = torch.log(p)
  print('log likelihood:', logp.item())
  nll = -logp
  print('negative log likelihood:', nll.item())
  nlls[i] = nll

print('=========')
print('average negative log likelihood, i.e. loss =', nlls.mean().item())
```
- gradient descent
```python
for k in range(20):
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean() # control the add same counts 
  print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  W.data += -50 * W.grad
```
- Prediction
```python
# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  out = []
  ix = 0
  while True:    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```
---
## 03-building makemore: MLP
- Refers
  - [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
  - https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb
  - https://colab.research.google.com/drive/1YIfmkftLrz6MPTOO9Vwqrop2Q5llHIGK?usp=sharing
  - A Neural Probabilistic Language Model [bengio03a](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- **Context is necessary**
  - use before 3 words —> the 4th word
  - Lookup table —> word -> vector
  - Share the vectors
- ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-04-26%209.53.05.png)
- Padding and before N words
```python
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words[:1]:
  print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix] # crop and append
```

```python
emma
... ---> e
..e ---> m
.em ---> m
emm ---> a
mma ---> .
```

- Lookup table and embeddings
![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-04-26%2010.08.07.png)
- `torch.cat(torch.unbind(emb, 1), 1)`
- `emb.view(-1, 6) == torch.cat(torch.unbind(emb, 1), 1)`
  - view is faster than cat
- Activation function
  - `h = torch.tanh(emb.view(-1, 6) @ W1 + b1)`
- Tanh and sigmoid
- ![](NN%20From%20zero%20to%20Hero/image%204.png)
  - Broadcasting (`+b1`)
    - Align on the right and create a fake dimension here.
- Loss
```python
for i in range(200000):
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))
  
  # forward pass
  emb = C[Xtr[ix]] # (32, 3, 10)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
  logits = h @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Ytr[ix])
  #print(loss.item())
  
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  #lr = lrs[i]
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  #lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10().item())

#print(loss.item())
```
- Search the leaning rate
```python
lre = torch.linspace(-3, 0, 10000)
lrs = 10**lre
for i in range(10000):
  
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))
  
  # forward pass
  emb = C[Xtr[ix]] # (32, 3, 10)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
  logits = h @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Ytr[ix])
  #print(loss.item())
  
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  lr = lrs[i]
#   lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  lri.append(lrs[i])
  stepi.append(i)
  lossi.append(loss.log10().item())

print(loss.item())

plt.plot(lri, lossi)
```
![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-04-26%2012.48.27.png)
- Low loss not means the model is better
  - Train set 80%
  - Validation set 10% (loss)
  - Test set 10%
  - Keep they have the same distribution
  - train loss is same as validation loss
- Risk of overfitting
- Small batch size
  - More noise in the training set
- Find bottleneck
  - embedding size
  - hidden layers
  - hidden layer nodes
- Embedding 
![](NN%20From%20zero%20to%20Hero/image%205.png)
- We often plot log(loss) not loss
---
## 04-Building makemore Part 3: Activations & Gradients, BatchNorm
- refers
  - https://colab.research.google.com/drive/1H5CSy-OnisagUgDUXhHwo1ng2pjKHYSN?usp=sharing
  - "Kaiming init" paper: [https://arxiv.org/abs/1502.01852](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa2VBTVprQk9LekViTE1NbFZEdzJWdk1iLWlsd3xBQ3Jtc0trY29xVlRRN21teE1HMUFqRVF3dVZKSUhEdTdFTWF3bVRZWHY0MGluWTFZYmpZMC1vMVdUT3Bfbm5LUnROdG9oNzYtZzhoNVhYX3o0cHB5UVB2c3FaYzdZb3MxeDVFNDNEUmFsdkRxXzRSdjd2TjhySQ&q=https%3A%2F%2Farxiv.org%2Fabs%2F1502.01852&v=P6sfmUTpUmc)
  - BatchNorm paper: [https://arxiv.org/abs/1502.03167](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqazJyei1OZ3dCY2xzZFJuYko4R01Xa2ZpTmYwd3xBQ3Jtc0tuM1A3SC1NbHp0dXpCWG9MdFpyM3AzakpQZURSaUxFNHVxT0ZJM2VrV3Jha2hsTDJYX2VJUEJuS1hjOUdJdUZUb2k0bTZLeTBfNG9FSl9TRXA1ZHJBMDNLYWdtS3pNS09YWTc5WlN5ZEVpa0pVQjFpdw&q=https%3A%2F%2Farxiv.org%2Fabs%2F1502.03167&v=P6sfmUTpUmc)
  - Bengio et al. 2003 MLP language model paper (pdf): [https://www.jmlr.org/papers/volume3/b...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqblBULW56d3RtU2RvYnpXV3FsWHZMNTR6YkhBQXxBQ3Jtc0tuY25EWlQzUGtGVTRkZU1zNXRwRGpTeTZiNTg0Z3VLYkxtSXJqVUMyRDRqN2NmLWNhVmlQU1NhNWZmbTR0VHF1VjQ1VXltX2RTczNWSWNnLVVVd2pwOFNnRkxIUkJxcEZLSjNXV2Q2VmNnMENqb3R0Zw&q=https%3A%2F%2Fwww.jmlr.org%2Fpapers%2Fvolume3%2Fbengio03a%2Fbengio03a.pdf&v=P6sfmUTpUmc)
  - Good paper illustrating some of the problems with batchnorm in practice: [https://arxiv.org/abs/2105.07576](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmR2MGRGczByMEgwb3BYOVZ0ckFzamJySmNiZ3xBQ3Jtc0ttZ3ZJLUVaeXduemNvRmtUM045ZTVxQXdjRWlXRF9hSVM0a0xwUlZXaXBPYmpqdmZISmV3LTc0MVNBeHI2UkxCSE9ScVpCcEZ6VVVFanc0WnVMTGRqanQxcHB5VElRVVpqNlFwQnNqODZyaWExaU9MRQ&q=https%3A%2F%2Farxiv.org%2Fabs%2F2105.07576&v=P6sfmUTpUmc)
- BatchNorm
```python
# BatchNorm parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))
```
- `torch.no_grad()`  -> disables gradient tracking
  - For validation or test
- Parameter init ->  Kaiming init
  - loss is not stable when with large parameters
    - loss -> inf
- Activation functions
  - **A dead neurons -> No activated -> No sample can make the neurons activated -> Dead**
  - **init would be NG**
  - ```python
    plt.figure(figsize=(20, 10))
    plt.imshow(h.abs() >= 0.99, cmap='gray', interpolation='nearest')
  - ![](NN%20From%20zero%20to%20Hero/image%206.png)
- Learning rate is very high -> Large gradient -> More dead neurons
  - Sigmoid, Tanh, ReLU
- Leaky Relu, Mazout, ELU
  - They had flat parts -> 
- ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-04-26%2020.47.06.png)
```python
## loss log

### original:
train 2.1245384216308594
val   2.168196439743042

### fix softmax confidently wrong:
train 2.07
val   2.13

### fix tanh layer too saturated at init:
train 2.0355966091156006
val   2.1026785373687744

### use semi-principled "kaiming init" instead of hacky init:
train 2.0376641750335693
val   2.106989622116089

### add batch norm layer
train 2.0668270587921143
val 2.104844808578491
```
- Gaussian before and after
  - w*1 -> Gaussian is expanding -> Larger scale
  - w*0.2 -> Gaussian is shrinking -> Smaller scale
  - We should care about the deeper W for them -> ReLU or PReLU
    - mean -> 0 and std -> (2/nl)^(0.5)
    - nl -> the number of neurons in the hidden layer 
    - [Function torch::nn::init::kaiming_normal_ — PyTorch main documentation](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ac8a913c051976a3f41f20df7d6126e57.html)
    - fan_mode = fan_in 
    - ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-04-26%2021.20.30.png)
    - ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-04-26%2021.25.13.png)
```python
x = torch.randn(1000, 10)
w = torch.randn(10, 200) # if we *0.2, the std will be smaller
y = x @ w

print(x.mean(), x.std())
print(y.mean(), y.std())
#tensor(0.0041) tensor(0.9947)
#tensor(0.0061) tensor(3.1653)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(x.view(-1).tolist(), 50, density=True);
plt.subplot(122)
plt.hist(y.view(-1).tolist(), 50, density=True);  
```
![](NN%20From%20zero%20to%20Hero/image%207.png)
- Make initialization of neural networks more not important
  - Residual module
  - Normalization
    - Barch, group 
  - Other optimazer
    - RMSprob, Adam
  - Care about input histogram and output histogram 
- Batch normalization
  - **It made it possible to train very deep neural nets quite reliably**
  - It basically just worked.
  - We do not want the ph-preact is too large because tang will ignore  them
  - We want them to roughly gaussian
  ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-04-26%2022.09.24.png)
  - `hpreact_bn = (hpreact - hpreact.mean(0, keepdims=True))/hpreact.std(0, keepdims=True)`
  - Every signal neuron network and its firing rate will be exactly unit gaussian on these batch samples.
    - At least of this batch
  - Sometime we do not want them must be the Gaussian
    - Distribution move around
    - Scale and shift -> 2 parameters -> we need to train them
    - ```python
      bngain = torch.ones((1, n_hidden))
      bnbias = torch.zeros((1, n_hidden))
      
      hpreact_bn = bngain*(hpreact - hpreact.mean(0, keepdims=True))/hpreact.std(0, keepdims=True) + bnbias
      ```
    - Batch normalization actually comes at a terrible cost
      - We need coupling these examples mathematically 
      - As kind of like regularizer
    - Layer normalization
    - Instance normalization
    - Group normalization
  ```python
    # BatchNorm layer
  # -------------------------------------------------------------
  bnmeani = hpreact.mean(0, keepdim=True)
  bnstdi = hpreact.std(0, keepdim=True)
  hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
  with torch.no_grad(): # no backward, only updates, tracking the paramer
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
  ```
  - bnmean = bnmean_running
  - bnstd = bnstd_running
  - we can use the running to instead of bnmean, and bnstd
  - Epsilon -> precent divided by zero
  - Batch normalization does not need bias.
    - we use bnbias to instead of bias
  - **Summary**
    - BN -> control the sprinkle of hidden layers
    - 2 Parameters: bngain and bnbias -> Need learn
    - 2 buffers: bnmean_running and bnstd_running -> maintaining
  - Restnet: A real example based batch normalized and very deep networks
    - Residual neural network -> image classification
  ![](NN%20From%20zero%20to%20Hero/image%208.png)
  - Bottleneck block
    - repeated and stacked up serially
    - bias is not used for batch normalization
  - Convolutions are basically linear layers except on patches
    - [Linear — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
  ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-04-29%2010.19.22.png)
  - [BatchNorm1d — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
  ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-04-29%2010.20.18.png)
  - momentum for running mean and std updates
- **Summary**
  - Understand the **activation and gradients** and their statistics in neural networks and 
  - this becomes increasingly important especially as you make your neural networks bigger larger and deeper.
  - Care about the **distribution of output layer** 
    - hockey stick losses and we can fix this to get a better loss at the end of training
    - because your training is not doing wasteful work
  - We also need to control the activations
    - **we do not want them squash to zero or explode to infinity**
  - We want **roughly gaussian activations** through the neural networks
  - Boost our results
- BN: not possible for much much deeper neural nets
  - because we sill have much different types of layers it become really **hard to precisely set** the weights 
  - and the biases in such a way that the activations are **roughly uniform** throughout the neural network.
  - Try to avoid it as much as possible 
- BN can control the statistics of the activations in a neural network
---
- RNN
  - A very deep networks: we need activation statistics
  - [torch.nn — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/nn.html)
```python
# Let's train a deeper network
# The classes we create here are the same API as nn.Module in PyTorch
class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
	return []
```
- networks
```python
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 100 # the number of neurons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((vocab_size, n_embd),            generator=g)
layers = [
  Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]
# layers = [
#   Linear(n_embd * block_size, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, n_hidden), Tanh(),
#   Linear(           n_hidden, vocab_size),
# ]

with torch.no_grad():
  # last layer: make less confident
  layers[-1].gamma *= 0.1
  #layers[-1].weight *= 0.1
  # all other layers: apply gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1.0 #5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
```
- Training -> Optimization 
```python
# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []
ud = []

for i in range(max_steps):
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
  
  # forward pass
  emb = C[Xb] # embed the characters into vectors
  x = emb.view(emb.shape[0], -1) # concatenate the vectors
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Yb) # loss function
  
  # backward pass
  for layer in layers:
    layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  with torch.no_grad():
    ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

  if i >= 1000:
    break # AFTER_DEBUG: would take out obviously to run full optimization
```
- Visualize output histogram
```python
# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out
    # t = layer.out.grad
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('activation distribution')
```
![](NN%20From%20zero%20to%20Hero/image%209.png)
![](NN%20From%20zero%20to%20Hero/image%2010.png)
```python
# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution');
```
![](NN%20From%20zero%20to%20Hero/image%2011.png)
```python
plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
plt.legend(legends);
```
- lr fast or slow
![](NN%20From%20zero%20to%20Hero/image%2012.png)
- Balancing a pencil on a finger
- Summary
  - Batch normalization
  - Diagnostic tools that you would use to understand whether your networks is in a good dynamically.
- Bottlenecked
  - Optimization
  - Context length
  - Network architectures
---
## 05-Building makemore Part 4: Becoming a Backprop Ninja
- refers
  - https://www.youtube.com/watch?v=q8SA3rM6ckI
  - https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb
  - https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing
  - Yes you should understand backprop: [](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGE1X2EtbFdWTTRWMV9uRmhWV0lENzdGVWdFd3xBQ3Jtc0tseDNncW0tQTJMUFZJekxsN2pvQmhOZmpLWjZmd05ueHBTWFJNaElVMTZPQTZMY2VhVWFpRURudTdzSVN6cTlQazhXRHQzcnU0QjQ2eFBCOFFWRUxDME9iODg4QTcwMHpxUGF4ck9Janl3TmltTDFINA&q=https%3A%2F%2Fkarpathy.medium.com%2Fyes-you-should-understand-backprop-e2f06eab496b&v=q8SA3rM6ckI)[/ yes-you-should-understand-backprop](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGE1X2EtbFdWTTRWMV9uRmhWV0lENzdGVWdFd3xBQ3Jtc0tseDNncW0tQTJMUFZJekxsN2pvQmhOZmpLWjZmd05ueHBTWFJNaElVMTZPQTZMY2VhVWFpRURudTdzSVN6cTlQazhXRHQzcnU0QjQ2eFBCOFFWRUxDME9iODg4QTcwMHpxUGF4ck9Janl3TmltTDFINA&q=https%3A%2F%2Fkarpathy.medium.com%2Fyes-you-should-understand-backprop-e2f06eab496b&v=q8SA3rM6ckI)
  - BatchNorm paper: [https://arxiv.org/abs/1502.03167](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa1dyVzNONURpVml4Z2ZjbGJERXdEclZjWUVoQXxBQ3Jtc0trZUN1Qmc2dGt0RHhTRGhzQThHMjQtX3ZxaFhLQk8tY1ZzdkIwdjRZbjZxWUxwSVFKUGVvZkl1MlFTclpidTJwMWQ5MGdMSWJIMHJUVjNzZ0NkZDEwYWk2dHYyd0ZkX2lvbG5sazZSb3dEUU04UXlHYw&q=https%3A%2F%2Farxiv.org%2Fabs%2F1502.03167&v=q8SA3rM6ckI)
  - Bessel’s Correction: [http://math.oxford.emory.edu/site/mat...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbkdrYkptUmtKQThTcWZZM25IQnB3U1FrMXI0d3xBQ3Jtc0trYUJRQ0pfUFNxY0w3aXU1ZEhyV2pWUlFURjNwMlZTT196dHV5X0VXLVBJbXB4LW9rclM5cmdkblh5VWE1SVNvUTBXdEltbHBZUUZxLWJTV2NDSEJkTHN1SmNQYmRmOFlXdVlUcGx1VEhPVkdPRzFuaw&q=http%3A%2F%2Fmath.oxford.emory.edu%2Fsite%2Fmath117%2FbesselCorrection%2F&v=q8SA3rM6ckI)
  - Bengio et al. 2003 MLP LM [https://www.jmlr.org/papers/volume3/b...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa1pHOW5EVlJFZWlsSEdOd09jT1p6bXp2Sk5LQXxBQ3Jtc0ttLUxDLXJqOUl6WW44Y2RwUk5QWTB5cHJKX0taR2IyV3czQ2t5N1NucHNGQkQxYWpqWEo1ME5VSDZHRUhiUUMxWHpGQTZFRG55eVd0UFRjeEYtM2VOcV9NaGliMTJWV2NacW1aODVIazdabHR2MktTMA&q=https%3A%2F%2Fwww.jmlr.org%2Fpapers%2Fvolume3%2Fbengio03a%2Fbengio03a.pdf&v=q8SA3rM6ckI)
- **Back-propagation is that it is a leaky abstraction.**
- **TLDR**: 
  - if you’re using **sigmoids** or **tanh** non-linearities in your network and you understand back propagation 
  - you should always be nervous about 
    - making sure that the **initialization** doesn’t cause them to be fully saturated.
  - ![](NN%20From%20zero%20to%20Hero/image%2013.png)
  - ![](NN%20From%20zero%20to%20Hero/image%2014.png)
- Debugging my networks
- ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-04-30%2020.57.58.png)
```python
# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
```

```python
dlogprobs = torch.zeros(batch_size, vocab_size) # 32 * 27
dlogprobs[range(n), Yb] -= 1/batch_size # 32*27
dprobs = dlogprobs * (probs)**(-1)
dcounts_sum_inv = dprobs * counts
dcounts_sum_inv = dcounts_sum_inv.sum(1, keepdim=True)
dcounts_sum = -dcounts_sum_inv * counts_sum**(-2)

dcounts = counts_sum_inv * dprobs
dcounts += torch.ones_like(counts) * dcounts_sum

dnorm_logits = dcounts * counts
dlogit_maxes = (-dnorm_logits * logit_maxes).sum(1, keepdim=True)
dlogits = dnorm_logits.clone() - F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes

dh = dlogits @ torch.transpose(W2, 0, 1)
dW2 = torch.transpose(h, 0, 1) @ dlogits
db2 = dlogits.sum(0, keepdim=True)
dhpreact = (1-h**2)*dh

dbngain = (dhpreact * bnraw).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)

dbnraw = dhpreact * bngain
dbnvar_inv = bndiff * (-0.5) * (bndiff+1e-5)**(-1.5) * dbnraw


```
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202024-05-02%2017.57.10.png)
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202024-05-02%2018.12.55.png)
---
## 06-Building makemore Part 5: Building a WaveNet
- refers
  - makemore on github: [https://github.com/karpathy/makemore](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbEVzV2hKRmo3eDZyUVowVVRfVFJBYzh1TWh6Z3xBQ3Jtc0ttbEd4OXRhS1VZd1dVMWFkX2xISjZHemtEczVQZUdRdE9mYWEyQV9wLUMxREVxdHRpTlhuUEVBNTlldlhtajBaZFBkcE9wZzJKdi1yRWtnZkVSS1lFMUZNWnRXNDJ6dHZVTndkOXBONjhpSjM2SDdxbw&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fmakemore&v=t3YJ5hKiMQ0)
  - jupyter notebook I built in this video: [https://github.com/karpathy/nn-zero-t...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbXQ5eWJRX0NYQVZyWGNZckVqM3lNY1VFcTZEd3xBQ3Jtc0trOEZ5MXNET21qLXNNU0xmbTF3bFJab083ZGhzM3hxWVVVczlqc1JXOVVQLTdyd09ZSzIzVXZXenZZbEh4UFVOU0VvektuZUFyQlNESEk5UGo2d2lheTJUdmxIV2I4Mm5VM21QM1B6ZzBYNmM1QnVrWQ&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fnn-zero-to-hero%2Fblob%2Fmaster%2Flectures%2Fmakemore%2Fmakemore_part5_cnn1.ipynb&v=t3YJ5hKiMQ0)
  - collab notebook: [https://colab.research.google.com/dri...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGdaU09mWmpNS0ZwTkhScEpZSlZCTnlmNV81d3xBQ3Jtc0trRnV4Nmg3bTBYcU5mRDBwUFUwQXJIVlhzYi1ETGlSWE0teTJHWUttVUk4RDVpY21FY3FaSnY3WERpN2lnNklEWUdBNmpwLVJuMThKcnlhZ3RCVkN6T3NHeUpmX3NER1NKUlg3d0NEbWZ5S2VyWXlWNA&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1CXVEmCO_7r7WYZGb5qnjfyxTvQa13g5X%3Fusp%3Dsharing&v=t3YJ5hKiMQ0)
  - WaveNet 2016 from DeepMind [https://arxiv.org/abs/1609.03499](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbFhpRG1nVE50ckk0WjJyNGJGQ19yZDJoR3M3Z3xBQ3Jtc0tuVnAwN2JQTndYY2piZWFPZHU2dEdYUGZsNUo3SVJ1VWZGRTQxZE0ySWtjUFNza2tDWm1ONk9DeTRXbnBiUXd3Y1hVNHdQZ0tISTloZUhaR3hNUG4wS1NVQlpaaTI3MDJXVy1pc19JTy1rVm9jYkNERQ&q=https%3A%2F%2Farxiv.org%2Fabs%2F1609.03499&v=t3YJ5hKiMQ0)
  - Bengio et al. 2003 MLP LM [https://www.jmlr.org/papers/volume3/b...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbjFzWTBNZmU0MFVwS3gwQ2xYNG1ZVHBPOW5TZ3xBQ3Jtc0tuNkhHWVU5aVhQdmhTWXk4Qi1NUnpDX0k2SG03S052YTk3NGZybXFzU2JYSE9sQlE2RUVpSWd3LVd0bm5kX2tDSDRKUGNDMkNUejdHTnpjQnVVSFRybHBEdW1ZeGhOMmhVUXlBdDdVRUkwQTVKLXB2cw&q=https%3A%2F%2Fwww.jmlr.org%2Fpapers%2Fvolume3%2Fbengio03a%2Fbengio03a.pdf&v=t3YJ5hKiMQ0)
- Wavenet
- ![](NN%20From%20zero%20to%20Hero/image%2015.png)
```python
class Linear:

  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # note: kaiming init
    self.bias = torch.zeros(fan_out) if bias else None

  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)

  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      if x.ndim == 2:
        dim = 0
      elif x.ndim == 3:
        dim = (0,1)
      xmean = x.mean(dim, keepdim=True) # batch mean
      xvar = x.var(dim, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]


class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []


class Embedding:

  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))

  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out

  def parameters(self):
    return [self.weight]


class FlattenConsecutive:

  def __init__(self, n):
    self.n = n

  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
    return self.out

  def parameters(self):
    return []


class Sequential:

  def __init__(self, layers):
    self.layers = layers

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out

  def parameters(self):
    # get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]
```
- [torch.nn — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/nn.html)
  - Containers
- dilated causal convolutional layers
  - Progressive fusion
```python
model = Sequential([
  Embedding(vocab_size, n_embd),
  FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, vocab_size),
])
```
- use `view` to change the dims
```python
for layer in model.layers:
    print(layer.__class__.__name__, ':', tuple(layer.out.shape))
```
- 2 notebooks
  - kick off experiments 
  - check the shapes
  - training and validation loss
  - hyper parameters searches
---
## 09-Let's build the GPT Tokenizer: Byte Pair Encoding
- refers
  - Google colab for the video: [Tokenization.ipynb](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)
  - GitHub repo for the video: minBPE [https://github.com/karpathy/minbpe](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVFobDFocS1td1M3X0R0TG90Wmdjd3hYRFRjZ3xBQ3Jtc0ttRWRUcTVkWGxvLTU4bkpic1dwX1YwemdxdGNuWnZGNC1ldXRMVGVtVS1TaHZWQVo0NTNpNUpPLUVRNWlsWER1MzREcnEyUzBFR3RfQmtmYTg1a05hX2Rhb2NTNWFRa2VMT1V6VGdvZlRGWmE4VUxCQQ&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fminbpe&v=zduSFxRajkE)
  - tiktokenizer [https://tiktokenizer.vercel.app](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa1puMnJuR3M5SHRmRkRXSGFrSWsxUDgzRnJmZ3xBQ3Jtc0tuWkhvTFpjNWhXeUxJcWUydUNRLUZtNjFJNFFhR3doYXI1R1owU1QySkt4Mk5nNWh4M05QX1ZmQlB5TmRIcV8yVlBtQWYta3BSaXlEWlpRdFBEREl3cTVuRWZ0Q3pCNHpKRXdHWXlvNDVMMFFRemtNcw&q=https%3A%2F%2Ftiktokenizer.vercel.app%2F&v=zduSFxRajkE)
  - tiktoken from OpenAI: [https://github.com/openai/tiktoken](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa19hQlhSSVRRLV94Z1lJc2VlUDlPOGlNa3dlZ3xBQ3Jtc0tsOHZBdVVOTC1Mb3hrVExyR1AxWkZvWFA0YVhPemcwSmVYcUxVUnVkV2MyX0J0YTM3ZC1IdERldFluanFNN2daeEVkdTJBWTZ4Z05MLWw3NGdCRE9nRHRMY3BXQ0MxVEdhbkpmcUJ2NXQyWXNUZ0pUZw&q=https%3A%2F%2Fgithub.com%2Fopenai%2Ftiktoken&v=zduSFxRajkE)
  - sentencepiece from Google [https://github.com/google/sentencepiece](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbXpfYUZkNnhDbTIteEJkLTNTbHBZS3hwYVFYZ3xBQ3Jtc0treGVNQlppQ1ZEQ3p6anlNOU1TOHA2WmNabHlnSENxWjlNdUFEaEI5VGVURVpxYXZOMGc3V3RkcXFldXZJbnB6STNPRTNRcHdFSW5RVGpKUXBNc1lUcFFKSExrWXluNUdTazhxdDhrV2dEek1IWTBGMA&q=https%3A%2F%2Fgithub.com%2Fgoogle%2Fsentencepiece&v=zduSFxRajkE)
  - Paper: [language_models_are_unsupervised_multitask_learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- minBPE
  - train the tokenizer vocabulary and merges on a given text
  - encode from text to tokens
  - decode from tokens to text.
### Tokenization
- A look up table for each token
- Encode text to id sequence
- Not dealing with the character level, we are dealing with on chunk level.
  - GPT2: 50257 tokens
  - Llama: 2 T tokens
- Tokens are the fundamental unit/ the atom of LLM
  - Translating strings or text into sequences of tokens
### Tokenization is at the heart of much weirdness of LLMs. Do not brush it off.
* Why can't LLM spell words? **Tokenization**.
* Why can't LLM do super simple string processing tasks like reversing a string? **Tokenization**.
* Why is LLM worse at non-English languages (e.g. Japanese)? **Tokenization**.
* Why is LLM bad at simple arithmetic? **Tokenization**.
* Why did GPT-2 have more than necessary trouble coding in Python? **Tokenization**.
* Why did my LLM abruptly halt when it sees the string "<|endoftext|>"? **Tokenization**.
* What is this weird warning I get about a "trailing whitespace"? **Tokenization**.
* Why the LLM break if I ask it about "SolidGoldMagikarp"? **Tokenization**.
* Why should I prefer to use YAML over JSON with LLMs? **Tokenization**.
* Why is LLM not actually end-to-end language modeling? **Tokenization**.
* What is the real root of suffering? **Tokenization**.
- **Space is one part of token chunk.**
  - Sometimes, they are different tokens.
![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202024-10-01%2017.22.29.png)
- Transformer paid attention to each tokens.
  - ignore and merge some useless tokens
  - make the token distribution more flat
- Unicode —> 150,000 characters —> a way to define lots of types of characters
  - `ord("h")` —> unicode —> live, keep changing —> utf-8/16/32 —> Wasteless space —> not suitable for long sequence
- Why we need a tokenization?
- Byte pair encoding
  - Basic idea: Same pattern encoding —> Make sequence smaller —> Modeling
  - `text.encode("utf-8")` —> Very long sequence —> Computation cost
  - `print(sorted(((v,k) for k, v in stats.items()), reverse=True)`
    - —> Token frequency
  - ![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202024-10-01%2018.03.16.png)
```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
# print(stats)
# print(sorted(((v,k) for k,v in stats.items()), reverse=True))
```
- Merge them
```python
top_pair = max(stats, key=stats.get)

def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))
# [5, 6, 99, 9, 1]

tokens2 = merge(tokens, top_pair, 256)
print(tokens2)
print("length:", len(tokens2))
```
- A hyper parameter:
  - the more steps we take,
  - the larger will be our vocabulary,
  - the shorter will be our sequence. —> What we want!
  - The parameter —> Search the best in the practice
  - GPT4 Token size is 100,000 
- Merge in loop
```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# ---
# Hyper parameters
vocab_size = 276 # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx


print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
# tokens length: 24597
# ids length: 19438
# compression ratio: 1.27X
```
- Note:
  - the Tokenizer is a completely separate, independent module from the LLM. 
  - It has its own training dataset of text (which could be different from that of the LLM), on which you train the vocabulary using the Byte Pair Encoding (BPE) algorithm. 
  - It then translates back and forth between raw text and sequences of tokens. 
  - The LLM later only ever sees the tokens and never directly deals with any text.
- Encoder and Decoder: A massive pre-processing step a stage.
  - Separate to the LLM training set
  - WE do not care about code or not code. —> Tokenizer can help us encode them.
  - E.g. Less Japanese data, —> Token will be larger —> Sequence will be less
  - Finite context length which it can work on in the token space
![](NN%20From%20zero%20to%20Hero/image%2025.png)
- Decode
  - Given a sequence of integers in the range [0, vocab_size], what is the text?
```python
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

print(decode([128]))
```
- Encode
```python
def encode(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2: # less than 2 means no need to merge
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

print(encode(""))

text2 = decode(encode(text))
print(text2 == text)
```
- Forced splits using regex patterns (GPT series)
  - Enforce rules for what parts of the text will never be merged for sure.
  - we use `regex` as `re`
  - https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53 —> inference code
  - Training code for the GPT2 tokenizer was never released.’
  - GPT4 changed the regular expression that they use to chunk up text
```python
import regex as re
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

print(re.findall(gpt2pat, "Hello've world123 how's are you!!!?"))
# ['Hello', "'ve", ' world', '123', ' how', "'s", ' are', ' you', '!!!?']
```
- GPT Tokens
  - inference only, no training code
```python
!pip install tiktoken # added for colab

import tiktoken

# GPT-2 (does not merge spaces)
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("    hello world!!!"))
# [220, 220, 220, 23748, 995, 10185]

# GPT-4 (merges spaces)
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("    hello world!!!"))
# [262, 24748, 1917, 12340]
```
- Tokenizer files
```python
!wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe
!wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json

import os, json

with open('encoder.json', 'r') as f:
    encoder = json.load(f) # <--- ~equivalent to our "vocab"

with open('vocab.bpe', 'r', encoding="utf-8") as f:
    bpe_data = f.read()
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
# ^---- ~equivalent to our "merges"
```
- Special tokens GPT2
  - 50,000 merges + 256 raw byte tokens + 1 specical tokens
```python
encoder['<|endoftext|>'] # the only special token in use for the GPT-2 base model value = 50256
```
- Extending special tokens like:
  - Special tokens. `<|im_start|>: 100264` `<|im_end|>: 100265`
  - FIM: fill in the middle —> paper titled Efficient training of language models to fill in the middle
- BPE md
  - [karpathy/minbpe: Minimal, clean code for the Byte Pair Encoding \(BPE\) algorithm commonly used in LLM tokenization.](https://github.com/karpathy/minbpe)
- Sentencepiece
  - We can do both training and inference tokenizers. (BPE, et)
  - Used in Llama and mistral series.
  - https://github.com/google/sentencepiece
  - **The big difference**:
    - sentencepiece runs BPE on the Unicode code points directly! 
    - It then has an option character_coverage for what to do with very very rare codepoints that appear very few times, 
    - and it either maps them onto an UNK token, or if byte_fallback is turned on, 
    - it encodes them with utf-8 and then encodes the raw bytes instead.
```python
import sentencepiece as spm

# train a sentencepiece model on it
# the settings here are (best effort) those used for training Llama 2
import os

options = dict(
  # input spec
  input="toy.txt",
  input_format="text",
  # output spec
  model_prefix="tok400", # output filename prefix
  # algorithm spec
  # BPE alg
  model_type="bpe",
  vocab_size=400,
  # normalization
  normalization_rule_name="identity", # ew, turn off normalization
  remove_extra_whitespaces=False,
  input_sentence_size=200000000, # max number of training sentences
  max_sentence_length=4192, # max number of bytes per sentence
  seed_sentencepiece_size=1000000,
  shuffle_input_sentence=True,
  # rare word treatment
  character_coverage=0.99995,
  byte_fallback=True,
  # merge rules
  split_digits=True,
  split_by_unicode_script=True,
  split_by_whitespace=True,
  split_by_number=True,
  max_sentencepiece_length=16,
  add_dummy_prefix=True, # we can know it is a start word or not
  allow_whitespace_only_pieces=True,
  # special tokens
  unk_id=0, # the UNK token MUST exist
  bos_id=1, # the others are optional, set to -1 to turn off
  eos_id=2,
  pad_id=-1,
  # systems
  num_threads=os.cpu_count(), # use ~all system resources
)

spm.SentencePieceTrainer.train(**options)

sp = spm.SentencePieceProcessor()
sp.load('tok400.model')
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
vocab
```
- We want to keep the raw data as much as possible 
  - Unrecognized and real would be shown as a punk, so we need `byte_fallback=True`.
- More tokens, more parameters for transformers, less sequence length
- Why vocab size cannot be infinity
  - Token embedding layers cannot be infinite 
  - Linear layer hidden nodes cannot be infinite 
  - More tokens —> more and more rarely in the training data —> fewer and fewer examples for each individual token —> we might worried that basically the vectors associated with every token will be undermined as a result. Because they do not come out too often and they do not participate in the forward backward pass.
  - More tokens —> Sequences will be sequences a lot —> we can attending more and more texts —> too large of chunks are being squished into single tokens and so the model just does not have as much of time to think per sort of some number of characters in the text. 
- What about we use a pre-train model and extend the vocab size?
  - Fine tuning, we also need fine tuning the tokenlizer
  - Introduce new parameters to train these new tokens into the architecture
  - Paper: Learning to Compress Prompts with Gist Tokens
    - Compressing prompt as tokens —> distill model to learn these new tokens —> only training the new token embeddings.
- Transformers not only the text, other modalities
  - Paper: Taming transformers for high-resolution image synthesis (CVPR 2021)
  - We only need to tokenized our data and do everything else identical in an identical manner 
  - Image —> chunks as numbers or soft tokens
  - OpenAI SORA —> Image as tokens on visual patches —> sequence
- Integer tokenization is insane (GPT2)
  - Arbitrary manner
  - Token are text chunk, not characters
  - unstable tokens 
- Tokenization data set is very different from the training data set for the actual language model
  - https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation
  - [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)
  - A ton of reddit data person post a lot.
    - the single individual token for that single Reddit user 
  - When we training the model, reddit data was not present. —> Some token never activated. —> unallocated memory —> the embedding of those tokens were never been trained —> these glitch token will generate some strange results
- vocab_size
  * Q: what should be vocab size?
  * Q: how can I increase vocab size?
  * A: let's see. Reminder: ~[gpt.py](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)~ from before.
- Final recommendation
  - Don't brush off tokenization. A lot of footguns and sharp edges here. Security issues. Safety issues.
  * Eternal glory to anyone who can delete tokenization as a required step in LLMs.
  * In your own application:
    * Maybe you can just re-use the GPT-4 tokens and tiktoken?
    * If you're training a vocab, ok to use BPE with sentencepiece. 
      * Careful with the million settings.
    * Switch to minbpe once it is as efficient as sentencepiece
* Also worth looking at:
  * [Huggingface Tokenizer](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftransformers%2Fmain_classes%2Ftokenizer)
  * I didn't cover it in detail in the lecture because the algorithm (to my knowledge) is very similar to `sentencepiece`, but worth potentially evaluating for use in practice.
---
## 10-ChatGPT-2
- Refers 
  - build-nanogpt GitHub repo, with all the changes in this video as individual commits: [https://github.com/karpathy/build-nan...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGVrcXh2WXNSX3B3MFduSFdYYzJQa1NaMm5mUXxBQ3Jtc0tsUEh5X1BlLVduNW5FYUYzMkpIUWFrRVBwVGMxV1FOamRZOXBDVTY5T1VteFFCRTZxdG1ya29kMlBKYzZEUDVRWUt1Ri1uTG9jZ0FVUmtwZVNtbkh6VFlaWm5KODNJSnFhdEROaTJmUFhHYTZyQ2lUaw&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fbuild-nanogpt&v=l8pRSuU81PU)
  - nanoGPT repo: [https://github.com/karpathy/nanoGPT](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa3c1ZVVyYVZpSTJjMU1aVzNHM0I3RG9KXzc5QXxBQ3Jtc0tsVWN0MDI3SDRhVU91UnRGN2xONjRjNjUwc1hOZnRCYzZQREpXRkVNdXNpbnJfY0JZbXBsUkJGNHJZRWRLMUVxTHg0VFhKXzBld3hDcy1oTVBmUk5OMDVGLVdlMWRlZHRMcm5KZ3gzdWVxV3Q2TXpQbw&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2FnanoGPT&v=l8pRSuU81PU)
  - llm.c repo: [https://github.com/karpathy/llm.c](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqazF2U2VVNW9DN3NxWVlTMVNmcDZHaUJ4TDR3Z3xBQ3Jtc0tsbXdVVDVLUDU5cWFMb0ZEN2pCRkJFNVRwSVd5bmltZ1dwOEM5bnpiRFBuUjV4Q09NdlMybmFZLXlhN3Bad3pYS0pYV3ZrM0hOTXNma0R4NEFhUXBqdE1lSVRpcVNfNkpVcDd4b1NOeWJPNHY4bnhrcw&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fllm.c&v=l8pRSuU81PU)
- Config
```python
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
- GPT class
```python
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
			# token embeddings weight
            wpe = nn.Embedding(config.block_size, config.n_embd),
			# position embeddings weight
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme --> Less parameters --> Faster
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
- Block
  - **Causal Self-Attention**
    - Attention before myself
    - we need mask after tokens
  - **Layer normalization**
    - Keep mean is 0 and std is 1 by more deep networks
  - **Residual model**
    - To prevent gradient disappear
```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # residual 
        x = x + self.mlp(self.ln_2(x))
        return x
```
- MLP
  - [GELU — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
    - estimation would be faster than cumulative distribution function for Gaussian
      - [Cumulative distribution function](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
      - ![](NN%20From%20zero%20to%20Hero/image%2021.png)
    - Smooth than Relu
      - Dead Rely neuron problem -> Zero gradient
```python
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
```
- Attention block
  - [torch.nn.functional.scaled_dot_product_attention — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
![](NN%20From%20zero%20to%20Hero/image%2016.png)
- Flash attention —> More fast
![](NN%20From%20zero%20to%20Hero/image%2022.png)
```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() 
		# batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
		# (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
		# (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
		# (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
		# flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
		# re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
```
- By hands
  - Never tokens in the future —> Causal is True
  - softmax
    - ![](NN%20From%20zero%20to%20Hero/image%2019.png)<!-- {"width":213.00000000000006} -->
```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf))
att = F.softmax(att, dim=-1)
y = att @ v # (B nh T T) * (B nh T hs) --> (B nh T hs)
```
- Forward
  - [Tiktokenizer](https://tiktokenizer.vercel.app/?model=gpt2)
    - split word 2 index (int)
```python
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```
- Sampling pool
  - [torch.gather — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.gather.html)
  - Take the only last logit
  - Take top 50 and Sampling
    - We will never sampling rare tokens
```python
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
```
- Parameters
  - [build-nanogpt/train_gpt2.py at master · karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L157-L177)
```python
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
```
- Step by step debug
- We also need to check the initialization or probability
  - logit —> Search a good start point
    - logit = log(p)-log(1-p)
- Optimizer
  - https://qiita.com/omiita/items/1735c1d048fe5f611f80
  - **最急降下法**: **全データ** を使って損失関数が最小値になるように、勾配を使ってパラメータ更新するよ。
    - w = w - lr * G
    - 无法并行计算，全部数据后更新参数
    - 容易陷入局部最小值（微分=0）  —> Random
  * **SGD**: **データ1つだけ** をサンプルして使うことで、最急降下法にランダム性を入れたよ。
    * 每次使用一个样本数据 更新参数（random selection）-> 防止陷入局部最小
    * 无法并行计算
  * **ミニバッチSGD**: **データを16個とか32個** とか使うことで並列計算できるようにしたSGDだよ。
    * 可以并行计算
    * Pathological Curvature —> slow to update parameters
  * **モーメンタム**: SGDに **移動平均** を適用して、振動を抑制したよ。
    * v = beta * v + (1-beta) * G
    * w = w - lr * v
  * **NAG**: モーメンタムで損失が落ちるように保証してあるよ。
    * v = beta * v + (1-beta) * G (w-beta*v)
    * w = w - lr * v
  * **RMSProp**: 勾配の大きさに応じて **学習率を調整** するようにして、振動を抑制したよ。
    * v = beta * v + (1-beta) * G^2
    * w = w - lr * G / sqrt(v + e)
  * **Adam**: **モーメンタム + RMSProp** だよ。今では至る所で使われているよ。
    * w = w - alpha * moment * G
    * ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-06-19%2021.52.40.png)
  * **ニュートン法**
    * **二階微分** を使っているので、ものすごい速さで収束するよ。
    * ただ計算量が膨大すぎて **実用されていない** よ。
    * w = w - L’ / L’’
  * AdamW
    * https://zenn.dev/bilzard/articles/adamw-demistified
    * https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    * そこで、AdamWではAlgorithm 2（[3]より一部改変して転載）のように、*勾配のスケーリング処理*と*重み減衰*という2つの処理を分離した形式(decoupled)で実装することを提案する。
    * 両者の計算過程は独立しているので、互いに干渉することがない。
    * ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-06-19%2021.52.54.png)
    * normalization on each gradient element individually speed up the optimization especially for LM
    * More faster than SGD
* **Start with a zero gradient**
  * test it on a single batch try to overfilling it!
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
	optimizer.zero_grad() # Necessary!!!
	logits, loss = model(x, y)
	loss.backward()
	optimizer.step()
	print(f"step {i}, loss: {loss.item()}")

```
- Dataloader
```python
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
```
- Transformer share the wet weight with lm head weight
  - saved 30% parameters
  ```python
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
  ```
- Control the growth of residual steam
  - `x += n**-0.5 * torch.randn(768)`
```python
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```
- **Speed up training**
  - Float 32 is OK for deep learning —> Performance
  - INT 8 for inference, not for training
  - GPU memory bandwidth —> Faster GB/s: Load data more faster
  - We can store more and more fast
  - Matrix multiplication
  - Max batch size of your GPU
    - 16 … 64
  - `torch.set_float32_matmul_precision('high')`
    - [torch.set_float32_matmul_precision — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)
    - More faster
  - We need auto cast
    - `with torch.autocast(device_type=device_type, dtype=torch.bfloat16):`
    - [Automatic Mixed Precision package - torch.amp — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/amp.html)
- **torch.compile**
  - `model = torch.compile(model)`
  - Cost compile time, but more faster
    - [Introduction to torch.compile — PyTorch Tutorials 2.3.0+cu121 documentation](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
  - NOTE: a modern NVIDIA GPU (H100, A100, or V100) is recommended for this tutorial in order to reproduce the speedup numbers shown below and documented elsewhere.
  - Torch compile to optimize the compute flow, data location et. al.
    - More faster —> Save time for data communication (HBM)
- Flash attention
  - https://qiita.com/jovyan/items/11deb9d4601e4705a60d
  - [Dao-AILab/flash-attention: Fast and memory-efficient exact attention](https://github.com/Dao-AILab/flash-attention)
  - [torch.nn.functional.scaled_dot_product_attention — PyTorch master documentation](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention)
    - 特に**causal**以外のattention_maskを与えた場合にはFlash Attentionは適用されません。
  - ![](NN%20From%20zero%20to%20Hero/image%2017.png)
  - online softmax trick
  - `y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention`
  - 27 faster than before
```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:,:,:T,:T]) == 0, float('-inf'))
att = F.sofmax(att, dim=-1)
y = arr @ v
```
- **Optimization**
  - Try to use 16, 32, 64, 128 … in your code.
    - Lots of kernels are written in terms of powers of 2, 
    - Lots of blocks of size is 16 or 64
  - Fix ugly numbers
    - Token size is 50304 = 128 * 393
    - Improved 4%
- **AdamW gradient clipping**
  - GPT3 paper is more detailed
    - Different to GPT2 is the context length from 1024 to 2048
    - Bigger data set and training more longer
    - More parameters
  - **We clip the global norm of the gradient at 1.0**
    - [torch.nn.utils.clip_grad_norm_ — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
    - `norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
    - **bad data batch** -> Shock your model -> to prevent your model more stable
      - We should monitor the norm value
- **Learning rate schedule**
  - Cosine decay learning schedule with warmup
  - ![](NN%20From%20zero%20to%20Hero/image%2018.png)
  - **Learning to ignore some useless tokens**
```python
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)
```
- **Optimizer config**
  - weight_decay
    - One type of normalization
    - More about weight decay
    - **fused —> More faster on Cuda**
      - Instead of loop updating these parameter 
      - Those kernels are fused into a single kernel -> Single time on all the parameters
```python
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
		# layer normal sacles and biases
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
```
- **Gradient accumulation**
  - Larger model --> Smaller learning rate and larger batch size
  - Large batch size will make my GPU explode, but we also want to use that batch size
    - **We need gradient accumulation -> To simulate the large batch size**
    - Care about the `loss = loss / grad_accum_steps`
  - **ddp_world_size -> GPU count**
    - DistributedDataParalled: [DistributedDataParallel — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
    - `nvidia-smi`
  - Master process —> do some special
    - Other process —> back forward
  - `torchrun --standalone --nproc_per_node=8 train_gpt2.py`
  - `from torch.nn.parallel import DistributedDataParallel as DDP`
    - Can dispatch communications 
    - `ddp.no_sync()` -> require_backward_grad_sync
      - We want it to be True in the final iteration for each batch
      - `model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)`
  - `dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)`
    - To take average of loss_accum from all sub tasks.
  - **To apply optimizer to ddp, we need a raw_model**
```python
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)
```
- DDP model
```python
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```
- **Why we need / grad_accum_steps**
```python
import torch

net = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.GELU(),
    torch.nn.Linear(32, 1)
)

torch.random.manual_seed(42)
x = torch.randn(4, 16)
y = torch.randn(4, 1)

net.zero_grad()
yhat = net(x)
loss = torch.nn.functional.mse_loss(yhat, y)
loss.backward()
print(net[0].weight.grad.view(-1)[:10])

# Gradient accum steps is 4
net.zero_grad()
for i in range(4):
    yhat = net(x[i])
    loss = torch.nn.functional.mse_loss(yhat, y[i]) / 4 # We need the 4!
    loss.backward()
print(net[0].weight.grad.view(-1)[:10])
```
- grad_accum
```python
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# For training
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
```
- Distributed Data Parallel (DDP)
```python
# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
```
![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-06-15%2011.50.45.png)
- **Data set**
  - gpt-2 
    - web text data, never released —> 40GB
  - gpt-3
    - extremely noise data and random subset of the internet, never released
  - red pajama data set
    - slim pajama 627B token —> cleaned and duplicated version of it
    - https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama
  - Fineweb
    - High quality common collection
      - How to collect these data
    - *15 trillion tokens*
    - https://huggingface.co/datasets/HuggingFaceFW/fineweb
    - Fineweb edu
      - sample -10BT —> Very closed to gpt2 performance
    - https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-06-15%2013.48.59.png)
- **Train validation sampling**
  - reset data loader
```python
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
```
- **Evaluation data set: hella swag**
  - https://huggingface.co/datasets/Rowan/hellaswag
    - smooth evaluation
  - domains
  - native form
    - ![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-06-15%2016.34.16.png)
    - pad the length and mask
    - check the probability of token and average them
      - [build-nanogpt/hellaswag.py at master · karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py)
    - Large model can see these options and select one.
- **log**
```python
# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass
```
![](NN%20From%20zero%20to%20Hero/%E6%88%AA%E5%B1%8F2024-06-15%2018.50.31.png)
- **Reason**
  - Trading data distribution is different
    - Not much math and code
  - Data not smooth, we need random sampling
  - break the document and shuffle them
- **Save checkpoint**
  - state_dict() -> Parameters
- Chart with it
  - We have to fine-tune it into the chat format.
  - Data set more conversational and a user assistant user assistant kind of structure
- **llm c**
  - got-2 and gpt-3 using cuda
  - more faster
  - https://github.com/karpathy/llm.c
  - less space
---
## RLHF
- https://huggingface.co/docs/trl/en/ppo_trainer
![](NN%20From%20zero%20to%20Hero/image%2020.png)

---
## Knowledge Distillation
- https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
- [Knowledge Distillation Tutorial — PyTorch Tutorials 2.4.0+cu121 documentation](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
### Knowledge distillation
- [Why] Knowledge distillation is a technique that enables
  - knowledge transfer 
    - from **large, computationally expensive models** 
    - to **smaller** ones without losing validity. 
  - This allows for 
    - **deployment on less powerful hardware**
    - making **evaluation faster**
    - **more efficient**
  - Applications of this technology can be found in devices such as drones or mobile phones. 
### Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- Data set 
```python
# Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

#Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
```
- Model and utility functions
  - Teach model and student model
```python
# Deeper neural network class to be used as teacher:
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Lightweight neural network class to be used as student:
class LightNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```
- Teacher and student relationships
  - Loss: Cross-entropy
    - https://qiita.com/ground0state/items/8933f9ef54d6cd005a69
![](NN%20From%20zero%20to%20Hero/image%2023.png)
- Necessary parameters
  * **model**: A model instance to train (update its weights) via this function.
  * **train_loader**: We defined our train_loader above, and its job is to feed the data into the model.
  * **epochs**: How many times we loop over the dataset.
  * **learning_rate**: The learning rate determines how large our steps towards convergence should be. Too large or too small steps can be detrimental.
  * **device**: Determines the device to run the workload on. Can be either CPU or GPU depending on availability.
* Train
```python
def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward() # 计算损失函数相对于每个参数的梯度，执行反向传播。
            optimizer.step() # 使用之前计算的梯度，按照指定的优化算法更新模型参数。

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad(): # 用于简化资源管理，使代码更加简洁和安全，避免资源泄漏。
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
```
- Train teacher network
```python
torch.manual_seed(42)
nn_deep = DeepNN(num_classes=10).to(device)
train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_deep = test(nn_deep, test_loader, device)
```
- Knowledge distillation run
  - **T**: 
    - Temperature controls the smoothness of the output distributions. 
    - Larger T leads to smoother distributions, thus smaller probabilities get a larger boost.
  * **soft_target_loss_weight**: 
    * A weight assigned to the extra objective we’re about to include.
  * **ce_loss_weight**: 
    * A weight assigned to cross-entropy. Tuning these weights pushes the network towards optimizing for either objective.
    * `torch.nn.KLDivLoss()`  —>  **pointwise KL-divergence**
    * https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    * ![](NN%20From%20zero%20to%20Hero/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88%202024-09-10%2016.15.51.png)
![](NN%20From%20zero%20to%20Hero/image%2024.png)
```python
def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
			# torch.nn.KLDivLoss()

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

# Compare the student test accuracy with and without the teacher, after distillation
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
```
## Laerning rate: Post-hov EMA
- [Post-hoc EMA - EMAの減衰パラメータの事後最適化](https://speakerdeck.com/yu4u/post-hoc-ema-emanojian-shuai-parametanoshi-hou-zui-shi-hua)
- 