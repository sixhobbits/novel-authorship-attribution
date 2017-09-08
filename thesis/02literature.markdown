Background {#chap:back}
==========

In this chapter, we will review prior literature relating to authorship attribution. We will further review prior literature that relates to the related fields of text classification and language modelling. Note that many of the articles that we make reference to are arXiv[^1] preprints and not all of them have been published in peer-reviewed journals. Due to the fast-moving nature of the field, we believe that this work should be taken as part of the prior literature even without having proven itself through peer-review.

#  Authorship Attribution

In this section we present a review of prior work that is closely related to our research on Authorship Attribution (AA). The two main sub-tasks of AA, as discussed before, are Authorship Identification (AI) and Authorship Verification (AV). We discuss each of these in turn. 

## Authorship Identification

Much prior work related to AI uses settings that are not common in real-world authorship attribution tasks [@luyckx2011scalability] talks about the scalability issues relating to authorship identification and criticises prior research on two main points. First, earlier work often uses a very small number of candidate authors for AI tasks (sometimes only two). Second, this work often uses very long texts (often each text is a full-length book). By contrast, in practical AA tasks there is often only a short fragment of text available, and a large number of candidate authors. @Luyckx states:

"authorship attribution ‘in the wild’ may entail thousands of candidate authors with often small sets of data or only very short texts, in substantially more topics, genres, and registers"

[@akimushkin2017role] A longer paper on text networks. Criticises older
studies, but still uses 10 books / author.

[@abbasi2008writeprints] extensively analyze stylometery Authorship
Attribution methods focussing on ‘cyberspace’. They use the Enron data
set.

[@somers2003authorship] look at some basic stylometry features such as
lexical richness and take special note of Alice Through the Needle’s
Eye, sometimes claimed to be the best Pistache.

## Authorship verification

[@luyckx2008authorship] describe why the verification task is more
interesting than the identification one. They use a high-school essay
corpus of 145 authors. [@koppel2014determining] shows how to use the
Imposter’s algorithm for verification. Shows MiniMax similarity.

[@halvani2016authorship] present results for authorship verification on
a several corpora, including PAN. They achieve a median accuracy of 0.7.

[@halvani2017authorship] evaluate simple and fast compression-based
models for authorship verification. They compare their system to Bagnall
and GLAD, and use the PAN dataset.

## Authorship style

We have discussed how it is possible to attribute a work to a specific
author based on specific stylistic features. However, the concept of
authorship style is not well-defined nor understood. Gatt and Krahmer in
an extensive survey on text generation, state

> What does the term ‘linguistic style’ refer to? Most work on what we
> shall refer to as ‘stylistic nlg’ shies away from a rigorous
> definition, preferring to operationalise the notion in the terms most
> relevant to the problem at hand.

For authorship attribution, style is closely associated with which
part-of-speech tags authors choose to use (an author might use many
adjectives), how they choose to punctuate (some authors are proud of
never having used a semi-colon in their lives; others use them
frequently), or how long they typically make their sentences. However,
style can be seen as a much broader concept. As discussed later,
@kabbara2016stylistic

## Related tasks

Detecting the language style of non native speakers
https://arxiv.org/pdf/1704.07441.pdf [@rudzewitz2016exploring] link
Authorship Attribution with Plagiarism Detection (and Short Answer
Assessment).

# General NLP and text classification

## Neural networks for authorship attribution

Work directly related to ours... [@shrestha2017convolutional] use CNNs
for authorship attribution on short texts (tweets). They cite Bagnall
and PAN, and discuss what the CNN learns.

[@kabbara2016stylistic] look at translating a document written one style
or genre to another, maintaining the content but transfering the style.
This is a proposal only that proposes using deep models for learning
style transfer of text.

[@yogatama2017generative] discuss and compare generative vs discrimative
LSTM models for classification tasks.

[@weissenborn2016neural] Show how most NLP tasks rely on two sequences
of text: either sequence to sequence, as in MT, or dual-sequence for
many classification tasks.

[@deng2015deep] compare generative and discriminative models, saying
that generative models can be better for discriminative tasks when
training data is limited as they can converge faster than generative
models

[@kestemont2014function] talks about why function words have generally
been regarded as important for authorship attribution; why they are more
useful for English than for other languages; and why character n-grams
perform so well.

[@chrupala2013text] are the first(?) to use character embeddings. They
use them to recognise code segments within text.

## Neural networks

[@raghu2016expressive] talk about the relation between the structure of
a neural network and the functions that it is able to compute. They
state that lower layers are more important and are more sensitive to
noise and optimizations, while higher layers can model exponentially
more complex functions.

[@spieckermann2015multi] does an extensive investigation of multi-task
and transfer learning using RNNs. Not much specific to language
modelling.

[@tiflin2012lstm] Show that LSTMs can be used for signature verification
(SatNac, UWC paper).

## Siamese Networks

[@bromley1993signature] Introduce the idea of Siamese networks for
Signature Verification.

[@naaman2017learning] use a siamese (RNN) network for learning a
pronunciation similarity function. Maybe I should feed the predictions
from generic and fine-tuned generative models into siamese network?

[@liu2013probabilistic] a master’s thesis on Siamese networks, mainly
wrt. images.

How can we model the Authorship verfication task? We can select N
authors, and model it as a classification problem, but this doesn’t
scale well to many authors, and is not a good fit (?) for the authorship
verification problem, where we don’t necessarily want to identify the
authors, but only want to know if two texts are written by the same
author. A similar problem is re-identification: We have one photo of a
person take as their canonical ID, and then want to know if another
photo taken of them is the same as the canonical one. Deep learning
using multiple channels (??) has been used to solve the image
Re-Identification problem [@zhu2017deep].

One-Shot learning is a way to use Neural Networks even when there is
only a small amount of training data available. Nice intro:
https://sorenbouma.github.io/blog/oneshot/ (using siamese networks);
paper http://www.cs.cmu.edu/ rsalakhu/papers/oneshot1.pdf; for
immitation learning (https://arxiv.org/abs/1703.07326)

[@baraldi2015deep] create a Siamese Network for learning scene detection
in videos. Deciding if two frames are from the same scene is related to
deciding if two texts are written by the same author. Interestingly,
they first teach the network to be able to identify objects, such as a
tree, by using Image Net, and then use the Siamese network to figure out
which features are indicative of a shared scene. This is simlar to first
training a generic language model, which learns language features, and
then using a siamese network to work out which features are indicative
of the same author.

[@hosseini2015similarity] use a Siamese network to help with OCR –
instead of doing full analysis of handwritten text in an image, they can
find similar images that have already been OCRd.

## Siamese networks for NLP

[@neculoiu2016learning] use a Siamese Network to match Job Titles on
similarity. For example, mapping “software architectural technician
Java/J2EE” to “Java Engineer”.

[@yin2015abcnn] use three models for Sentence Pairs, including
(Attention Based) Siamese Networks

[@mueller2016siamese] use siamese networks to achieve a new baseline in
a sentence similarity task.

[@hoffer2015deep] created a so-called “Triple Network”, which extends
the idea of a Siamese Network. It takes three inputs – X, X+ and X-,
where X and X+ are the same class and X- is a different class.i

## Transfer Learning and Style Transfer

[@shen2017style] use ‘style’ transfer where style refers to sentiment or
ciphertext. They use GaNs and VAEs. Nice discussion of previous style
transfer works.

[@zoph2016transfer] look at using Transfer Learning to solve data
scarcity issues

[@shin2016generative] use a ‘teacher-learner’ transfer model, in which
the student is trained on the output of the teacher(?).

[@riemer2017representation] IBM paper on multi-task learning and
transfer learning for text classification (Twitter sentiment). “The very
popular strategy of fine-tuning a neural network involves first training
a neural network on a source task and then using the model to simply
initialize the weights of a target task network up to the highest
allowable common representation layer”

[@lalor2017improving] talk about the fine-tuning a neural model by
adding supplemental data. They distinguish this from transfer learning,
where transfer learning uses data from a different task, while
fine-tuning uses more data from the same task.

[@mcgraw2016personalized] use an LSTM for speech recognition on mobile
devices, such as “Call Jacob”. They personalise language models by
adding the users’ contacts (bias the language models on the fly). This
can be seen as a crude version of transfer learning?

https://arxiv.org/pdf/1707.01161.pdf Use seq-to-seq models to translate modern english into shakespeare style english. they pretrain embeddings based on dictionaries.

## Language Modelling

Why is character-level language modelling so effective for Authorship
Attribution? The two most helpful features provided by character level
modelling are *affixes* and *punctuation*, both of which are better
captured by character-level models than word-based models.
[@sapkota2015not]

@gatt2017survey present a comprehensive modern survey (111 pages) on
Natural Langauge Generation. They include a section on generating
langauge in a specific genre or author style, and conclude that there is
much work still to be done in this area.

[@karpathy2015unreasonable2] shows how well an RNN can be used to model
different styles of language using only characters.

[@mikolov2012subword] look at using ‘subwords’ instead of characters.
They discuss the tradeoffs between word and character level language
modelling.

[@audhkhasi2017end] use a CNN to train character embeddings and then
feed these embeddings into a RNN to create a language model. They do
this as part of an end-to-end “ASR-free” keyword search from speech
system.

[@goga2013exploiting] show that language modelling is hard – to
deanonymize users on Yelp and Twitter they found that language was the
least useful feature, with location and timestamps of updates being much
better at identifying authors.

[@sutskever2011generating] Discuss character level RNNs in detail and
construct a large network (trained for five days on 8 GPUs).

[^1]: https://arxiv.org/
