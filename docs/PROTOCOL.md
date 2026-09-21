# Experimental Protocol

Working title: **Integrating External Evidence to Reduce Uncertainty in Link Prediction**

Status: Method protocol approved — official result-producing runs remain prohibited until the freeze gates in Decision 12 have passed.

## Decision 1: How external evidence is integrated

External textual evidence is integrated through entity embedding initialisation.

For entities with valid textual evidence, the text-derived vector is used as the initial entity embedding. Entity embeddings remain trainable and may change during knowledge graph training.

The core comparison will distinguish:

1. **Random initialisation**: entity embeddings are initialised without textual evidence.
2. **Correct-text initialisation**: each entity receives the vector derived from its own description.
3. **Shuffled-text initialisation**: entities receive valid text vectors belonging to other entities.

All conditions must otherwise use the same model definition, training budget, evaluation queries, and paired random components fixed by Decisions 6–11.

## Interpretation boundary

This design tests whether semantically informed entity initialisation affects link prediction and uncertainty estimation.

It does not test:

- frozen text representations;
- continuous text–graph fusion during training;
- alignment-loss methods;
- general multimodal evidence integration.

Any final conclusion must remain within this boundary.

## Rationale

This intervention is intentionally minimal. It isolates the effect of semantic entity information while avoiding additional fusion layers, alignment weights, and optimisation objectives that would introduce further uncontrolled design choices.

## Decision 2: What improvement in uncertainty means

Uncertainty is interpreted as the quality and reliability of predictive confidence, not merely as its numerical magnitude.

External evidence improves uncertainty estimation only when confidence more reliably reflects whether predictions are correct, without concealing a substantial deterioration in ranking performance.

Therefore:

1. A reduction in entropy is not sufficient evidence of improvement.
2. An increase in confidence is not sufficient evidence of improvement.
3. A low calibration error from a poorly performing or degenerate model is not sufficient evidence of improvement.
4. Uncertainty metrics must always be interpreted jointly with ranking metrics and the model's ability to distinguish correct from incorrect predictions.

The operational confidence definition and uncertainty metrics are fixed in Decision 9.

## Research questions

### RQ1: Link prediction performance

How does correct-text entity initialisation affect link prediction performance compared with random entity initialisation?

### RQ2: Semantic alignment

Are observed effects attributable to correct semantic alignment between entities and their descriptions, rather than merely to the distributional properties of pretrained text vectors?

This question will be examined by comparing correct-text initialisation with a fixed shuffled-text control.

### RQ3: Uncertainty quality

How does correct-text entity initialisation affect the reliability and usefulness of predictive confidence compared with random and shuffled-text initialisation, when interpreted jointly with ranking performance?

## Hypotheses

Directional hypotheses will be defined only after reviewing the theoretical and empirical literature. They will not be inferred solely from legacy pilot results.

## Decision 3: Official benchmark datasets

The official experiment will include three benchmark datasets:

1. **FB15k-237**
2. **WN18RR**
3. **CoDEx-M**

All three datasets are included before observing any new official result. A dataset must not be removed because its results are negative, inconsistent, or statistically insignificant.

### FB15k-237

FB15k-237 represents a multi-relational, real-world knowledge graph setting.

Its textual evidence will require an audited mapping from Freebase entity identifiers to an approved external text source. Mapping failures, ambiguities, and missing evidence must be recorded explicitly. No identifier or placeholder may silently be used as a description.

### WN18RR

WN18RR represents a lexical-semantic knowledge graph setting.

Its textual evidence will be derived from verified WordNet synset information. Because the graph and glosses originate from the same underlying lexical resource, this evidence will be described as auxiliary textual evidence rather than fully independent external evidence.

Entity coverage must be evaluated over the complete benchmark vocabulary, not only the training entities.

### CoDEx-M

CoDEx-M represents a Wikidata-based, real-world knowledge graph setting with benchmark-aligned textual metadata.

The official CoDEx resources provide aligned Wikidata descriptions and Wikipedia extracts. Missing or empty fields remain possible and must be audited rather than silently replaced.

### Cross-dataset interpretation

The three datasets differ simultaneously in graph structure, semantic domain, and evidence provenance.

Therefore, cross-dataset differences may be discussed as evidence of context sensitivity, but they must not be interpreted as a controlled causal comparison of text sources.

### Reporting commitment

- The same core intervention and control logic will be applied to all three datasets.
- All three datasets will be reported regardless of result direction.
- Dataset-specific preprocessing must be documented and tested.
- Official training is prohibited until every dataset passes its pre-defined data and evidence quality gates.
- If a dataset cannot pass a quality gate, any exclusion requires a documented protocol amendment made before observing model results.

## Decision 4: Knowledge graph embedding models

The official experiment will use two knowledge graph embedding models:

1. **TransE**
2. **DistMult**

### Model roles

TransE represents a translational, distance-based scoring family. It models a relation as a translation from the head entity representation towards the tail entity representation.

DistMult represents a multiplicative, bilinear scoring family. It scores a triple using an element-wise interaction between the head, relation, and tail representations.

Using these two models allows the experiment to test whether the effect of text-based entity initialisation depends on two structurally different but simple scoring mechanisms.

### Comparability requirements

- Both models will use real-valued entity embeddings.
- Both models will use the same entity embedding dimensionality.
- The same text-derived entity vectors will be used in the corresponding correct-text and shuffled-text conditions.
- No model-specific projection or fusion network may be added.
- Entity embeddings remain trainable in both models.
- Within each dataset–model block, the three initialisation conditions must use the same approved training and evaluation procedure.
- Hyperparameters may differ between dataset–model blocks only according to the pre-defined tuning protocol.

### Interpretation boundary

The experiment does not aim to establish state-of-the-art link prediction performance.

TransE may have limited capacity for complex one-to-many, many-to-one, and many-to-many relations.

DistMult uses a symmetric scoring function with respect to head and tail entities and therefore has limited capacity for asymmetric relations.

These limitations are part of the model comparison and must be acknowledged when interpreting dataset-specific results.

### Resolved implementation choices

The remaining model implementation choices are fixed by later decisions:

- TransE distance norm is selected only through the Random-baseline hyperparameter procedure in Decision 10;
- entity and relation initialisation, vector normalisation, and scaling are fixed in Decision 6;
- training-time embedding constraints and regularisation are fixed in Decision 8;
- training loss and negative sampling are fixed in Decision 8.

## Decision 5: Text evidence sources and encoder

All official text evidence must come from fixed, auditable source artifacts.

Live API responses must not be used directly in official experiments. Source files, repository revisions, parsing rules, coverage statistics, and SHA-256 hashes must be recorded before text encoding.

English is the official evidence language for all three datasets.

### FB15k-237 evidence

The candidate source is the text resource released by the KG-BERT authors for FB15k-237.

The following source fields will be audited:

- Freebase MID to entity name;
- Freebase MID to entity description.

For each entity, the evidence string will be constructed as `name + ". " + description`.

If the description is missing but a valid human-readable name exists, the evidence is explicitly classified as `name_only`.

If both the name and description are missing, empty, identifier-like, or invalid, the evidence pipeline must fail.

The source revision and file hashes must be pinned. Coverage must be evaluated over the complete benchmark vocabulary.

No live Wikidata or Wikipedia lookup may silently replace a missing FB15k-237 description.

### WN18RR evidence

The authoritative lexical source is Princeton WordNet 3.0.

A WordNet synset is uniquely identified by the combination of its offset and part of speech. An offset alone must not be treated as a unique synset identifier.

The WN18RR evidence pipeline must:

1. obtain the part of speech from a pinned, entity-aligned mapping artifact;
2. resolve the resulting `(offset, part of speech)` in Princeton WordNet 3.0;
3. require exactly one resolved synset;
4. cross-check the aligned mapping text against the resolved WordNet entry;
5. construct the evidence from the synset lemma or lemmas and gloss.

The evidence string will be constructed as `lemma(s) + ". " + gloss`.

Zero matches, multiple matches, empty glosses, version mismatches, or inconsistent mapping text must terminate the pipeline.

Coverage must be evaluated over the complete benchmark vocabulary, not only entities observed in training.

### CoDEx-M evidence

The authoritative source is the official CoDEx English entity metadata.

The evidence string will be constructed as `label + ". " + Wikidata description`.

If the description is missing but a valid human-readable label exists, the evidence is explicitly classified as `label_only`.

If both fields are missing, empty, identifier-like, or invalid, the evidence pipeline must fail.

Wikipedia page extracts are not part of the core experiment. They may be discussed as a possible extension, but they must not be introduced after observing official results.

### Evidence audit table

Before encoding, every dataset must produce a one-row-per-entity audit table containing at least:

- dataset;
- entity identifier;
- entity index;
- source artifact;
- source identifier;
- mapping status;
- evidence status;
- evidence text;
- character length;
- token length;
- evidence SHA-256.

The table must have exactly one row for every entity in the complete evaluation vocabulary.

### Text encoder

The official text encoder is `sentence-transformers/all-MiniLM-L6-v2`.

The encoder is frozen and used only to generate entity initialisation vectors. It is not fine-tuned during knowledge graph training.

The encoder produces 384-dimensional real-valued vectors. Therefore:

- the entity embedding dimension is fixed to 384 for TransE and DistMult;
- no learned or random projection layer is permitted;
- random, correct-text, and shuffled-text conditions all use 384-dimensional entity embeddings.

Before encoding, the following must be pinned and recorded:

- Hugging Face model revision;
- Sentence Transformers version;
- Transformers version;
- tokenizer configuration;
- maximum sequence length;
- pooling configuration;
- truncation behaviour.

The number and proportion of truncated evidence texts must be reported for every dataset.

### Scaling cross-reference

Text-vector normalisation and the matched random and relation initialisation scales are fixed in Decision 6.

## Decision 6: Initial embedding normalisation and scale

This decision defines the entity and relation embedding state at the start of training, denoted by \(t=0\).

Training-time norm handling and regularisation are fixed separately in Decision 8.

### Text-derived entity initialisation

The frozen text encoder produces one raw 384-dimensional vector \(v_e\) for every entity \(e\).

The raw text vectors must be preserved as a separately hashed artifact before any transformation.

Each valid text vector is explicitly L2-normalised before it is used to initialise the knowledge graph embedding model:

\(E_e^{(0)} = v_e / \lVert v_e \rVert_2\).

Therefore, every text-derived entity embedding has unit L2 norm at \(t=0\).

A raw vector containing a non-finite value or having an L2 norm less than or equal to \(10^{-12}\) must terminate the pipeline.

No centring, whitening, PCA, learned projection, random projection, or dimension-specific standardisation is permitted.

The normalised text embedding matrix must be stored as a separately hashed artifact.

### Correct-text initialisation

In the correct-text condition, each entity receives the normalised vector produced from its own audited evidence text:

\(E_e^{(0)} = \operatorname{normalise}(v_e)\).

### Shuffled-text initialisation

In the shuffled-text condition, entities receive the same set of normalised text vectors used in the correct-text condition, but with entity-to-vector assignments permuted.

Consequently, the correct-text and shuffled-text conditions must contain exactly the same multiset of initial vectors and vector norms.

The precise permutation procedure, derangement requirement, and shuffle-seed derivation are fixed in Decisions 7 and 11.

### Random entity initialisation

For each entity in the random condition, a 384-dimensional vector is sampled from an independent standard normal distribution:

\(z_e \sim \mathcal{N}(0, I_{384})\).

Each sampled vector is then L2-normalised:

\(E_e^{(0)} = z_e / \lVert z_e \rVert_2\).

Therefore, random, correct-text, and shuffled-text entity embeddings all have unit L2 norm at \(t=0\).

The random generator, experimental seed, and resulting initialisation artifact must be recorded.

### Relation initialisation

Relations do not receive textual evidence.

Each 384-dimensional relation vector is sampled from an independent standard normal distribution and L2-normalised to unit norm at \(t=0\).

Within each paired dataset–model–seed comparison, the random, correct-text, and shuffled-text conditions must receive identical copies of the same initial relation matrix.

Relation initialisation may differ between experimental seeds but must not differ between the three initialisation conditions belonging to the same paired comparison.

### Interpretation of comparisons

The three initialisation conditions support the following interpretations:

- correct-text minus random estimates the total effect of introducing correctly aligned text-derived initialisation;
- correct-text minus shuffled-text isolates the effect of correct entity–text semantic alignment;
- shuffled-text minus random estimates effects associated with the geometry and distribution of pretrained text vectors without correct semantic alignment.

The shuffled-text comparison is necessary because L2 normalisation equalises vector norms but does not make pretrained text vectors isotropic or distributionally identical to random unit vectors.

### Trainability boundary

All entity and relation embeddings remain trainable after initialisation.

This decision does not require the embeddings to remain unit-normalised during training.

No post-update normalisation or norm projection is used. The batch-local L2 regularisation rule is fixed in Decision 8.

### Initialisation audit

Before any official training run, the initialisation audit must record at least:

- dataset, model, condition, and experimental seed;
- matrix shape and numerical data type;
- random generator and seed information;
- minimum, maximum, mean, and standard deviation of raw text-vector norms;
- maximum absolute deviation from unit norm after normalisation;
- entity and relation matrix SHA-256 hashes;
- confirmation that paired conditions use identical relation initialisation;
- confirmation that correct-text and shuffled-text use the same multiset of text vectors.

All normalised vectors must be finite. Their L2 norms must equal 1 within a pre-specified absolute tolerance of \(10^{-6}\).

## Decision 7: Fixed shuffled-text control

The shuffled-text condition is an entity initialisation control. It is not a form of training-time data augmentation.

Its purpose is to preserve the geometry and distribution of the text-derived vectors while removing their correct semantic alignment with entities.

### Permutation unit

One shuffled-text permutation must be generated for every dataset–replicate combination.

The permutation must be generated before training and remain unchanged throughout all epochs, checkpoints, validation evaluations, and test evaluations belonging to that replicate.

Different replicate seeds use independently generated permutations.

For the same dataset and replicate seed, TransE and DistMult must use the same permutation. The permutation is therefore a property of the dataset replicate rather than of the KGE model.

A permutation must never be regenerated or selected in response to training or evaluation results.

### Permutation population

The permutation is applied to the complete evaluation vocabulary in canonical entity-index order.

It is generated only after:

1. the complete entity vocabulary has been frozen;
2. the evidence audit has passed;
3. every entity has exactly one valid evidence record;
4. the correct-text embedding matrix has been encoded and L2-normalised.

The shuffled-text matrix is created by permuting the rows of the normalised correct-text matrix.

### Validity constraints

Every permutation must be a bijection. Therefore:

- every target entity receives exactly one source vector;
- every source vector is used exactly once;
- the shuffled-text and correct-text conditions contain exactly the same multiset of vectors.

For every target entity \(e\), the assigned source entity \(\pi(e)\) must satisfy:

- \(\pi(e) \neq e\);
- the source evidence SHA-256 differs from the target evidence SHA-256;
- the source normalised-vector row hash differs from the target normalised-vector row hash.

These constraints prevent self-assignment and exact-content-equivalent assignment.

The procedure does not attempt to remove approximate semantic similarity between different descriptions. Such a rule would require an additional subjective similarity threshold and is outside the minimal protocol.

### Permutation generation

Candidate permutations are generated using a dedicated pseudo-random generator and a separately recorded shuffle seed.

The shuffle generator must use an RNG stream that is independent of:

- random entity initialisation;
- relation initialisation;
- batch ordering;
- negative sampling;
- any other training randomness.

A candidate is accepted only if all validity constraints pass.

Invalid candidates are rejected in full. They must not be manually repaired through selected swaps.

If no valid permutation is produced after 1,000 documented attempts, the pipeline must terminate rather than relax the constraints.

The official replicate seeds and shuffle-seed derivation are fixed in Decision 11.

### Cross-condition control

Within a dataset–replicate comparison:

- correct-text uses the entity's own normalised text vector;
- shuffled-text uses the vector assigned by the fixed valid permutation;
- random uses an independently generated random unit vector.

Correct-text and shuffled-text must use identical encoder artifacts, entity ordering, dimensionality, numerical data type, and vector multiset.

Only the entity-to-text alignment may differ between these two conditions.

### Permutation artifact

Every accepted permutation must be saved before training as a versioned TSV or equivalent tabular artifact containing at least:

- dataset;
- replicate identifier;
- shuffle seed;
- target entity index;
- target entity identifier;
- target evidence SHA-256;
- target normalised-vector row hash;
- source entity index;
- source entity identifier;
- source evidence SHA-256;
- source normalised-vector row hash;
- generation attempt number.

The complete permutation artifact must receive its own SHA-256 hash.

The audit must confirm:

- complete target coverage;
- complete source coverage;
- unique target indices;
- unique source indices;
- zero self-assignments;
- zero equal-evidence-hash assignments;
- zero equal-vector-hash assignments;
- equality of the correct-text and shuffled-text vector multisets.

### Statistical interpretation

Because each replicate uses a different valid permutation, seed-level variation in the shuffled-text condition includes both training stochasticity and variation arising from the random semantic misalignment.

The replicate remains the unit of paired statistical comparison.

No additional permutation may be selected because it produces more favourable, less favourable, or more interpretable results.

## Decision 8: Training objective, negative sampling, and training-time regularisation

All official models are trained only on triples belonging to the training split.

Validation and test triples must not be introduced as positive training examples, inverse training examples, or sources of training-time filtering information.

Higher model scores always indicate that a triple is considered more plausible.

### Pairwise ranking objective

TransE and DistMult use the same pairwise logistic ranking objective.

For a positive triple score \(s_i^+\) and each corresponding negative triple score \(s_{ij}^-\), the ranking loss is:

\(L_{\mathrm{rank}} = \frac{1}{BK}\sum_{i=1}^{B}\sum_{j=1}^{K}\operatorname{softplus}(s_{ij}^- - s_i^+)\),

where:

- \(B\) is the number of positive triples in the batch;
- \(K\) is the number of negative triples generated per positive triple;
- \(\operatorname{softplus}(x)=\log(1+\exp(x))\).

The loss decreases when a positive triple receives a higher score than its corresponding negative triples.

The objective does not treat uncalibrated KGE scores as probabilities and does not introduce a margin hyperparameter.

The value of \(K\) remains a validation-selected hyperparameter and must be locked before official comparisons.

### Uniform negative sampling

For every negative sample, exactly one side of the positive triple is corrupted.

The sampler must:

1. choose head corruption with probability 0.5 and tail corruption with probability 0.5;
2. sample the replacement entity uniformly from entities appearing in the training split;
3. sample with replacement;
4. reject a replacement equal to the original entity;
5. reject any generated triple that appears in the training split;
6. repeat until a valid negative triple is obtained.

Validation and test triples must not be consulted when rejecting training negatives. This prevents validation or test graph structure from influencing training.

If a valid negative cannot be produced after 1,000 documented attempts, the pipeline must terminate.

Repeated negative triples are permitted because sampling is performed with replacement. Their frequency must not depend on model scores.

### Excluded sampling and augmentation procedures

The core experiment does not use:

- Bernoulli relation-specific corruption;
- self-adversarial negative sampling;
- score-dependent negative sampling;
- in-batch negatives;
- hard-negative mining;
- reciprocal or inverse-relation training augmentation.

These procedures are excluded to keep the sampling distribution independent of the model condition and to minimise additional design variables.

### Paired stochastic training

Within each dataset–model–replicate comparison, the random, correct-text, and shuffled-text conditions must use identical:

- positive training triples;
- epoch-level batch order;
- head-versus-tail corruption choices;
- replacement entity samples;
- accepted negative triples.

Batch ordering and negative sampling must use dedicated RNG streams that are independent of:

- entity initialisation;
- relation initialisation;
- shuffled-text permutation generation;
- model parameter updates.

Negative samples must not depend on current or previous model scores.

Every run must record a streaming SHA-256 hash of its positive-batch order and accepted negative-triple sequence.

The three paired conditions must have matching sequence hashes. A mismatch invalidates the paired comparison and must terminate the affected official run group.

### Batch-local L2 regularisation

No entity or relation embedding is forcibly projected, clamped, or re-normalised after an optimiser update.

Instead, the training objective includes explicit batch-local L2 regularisation:

\(L = L_{\mathrm{rank}} + \lambda L_{\mathrm{reg}}\).

Let \(U_E\) be the unique entity embeddings referenced by the positive and accepted negative triples in the current batch, and let \(U_R\) be the unique relation embeddings referenced in that batch.

The regularisation term is:

\(L_{\mathrm{reg}} =
\frac{1}{|U_E|}\sum_{e\in U_E}\lVert E_e\rVert_2^2
+
\frac{1}{|U_R|}\sum_{r\in U_R}\lVert R_r\rVert_2^2\).

The coefficient \(\lambda\) is selected using only Random-baseline validation results under Decision 10.

For a given dataset–model configuration, the selected \(\lambda\) must remain identical across random, correct-text, and shuffled-text conditions.

Optimiser-level weight decay must be set to zero. This prevents an additional implicit regularisation mechanism from affecting embeddings that were not referenced in the current batch.

### Training-unseen entities

Entities that do not appear in the training split are excluded from the negative-sampling replacement population.

Because they are absent from positive training triples, negative samples, and batch-local regularisation, their entity embeddings must remain unchanged from \(t=0\) until evaluation.

Every official run must verify this property by comparing the initial and final rows of all training-unseen entity embeddings.

Results involving only training-seen endpoints and results involving training-unseen entities must be reported as mandatory secondary strata under Decision 9.

### Hyperparameter cross-reference

This decision fixes the objective form, sampling distribution, filtering boundary, and regularisation mechanism. The number of negatives \(K\), regularisation coefficient \(\lambda\), learning rate, batch size, epoch budget, and TransE distance norm are selected only through the validation-only procedure in Decision 10. Optimiser settings are fixed in Decision 10.

## Decision 9: Ranking and uncertainty evaluation

Evaluation is performed at the query level.

All official evaluation code must use the same scoring functions as training and must treat higher scores as more plausible predictions.

Validation results may be used only for hyperparameter selection and post-hoc calibration. Test results must not influence either process.

### Evaluation queries

Every validation or test triple \((h,r,t)\) produces two evaluation queries:

1. tail prediction: \((h,r,?)\), with \(t\) as the target entity;
2. head prediction: \((?,r,t)\), with \(h\) as the target entity.

The candidate set is the complete frozen evaluation vocabulary of the corresponding dataset.

The candidate vocabulary must not change between models, initialisation conditions, replicates, head prediction, tail prediction, or reporting strata.

### Split-specific filtered evaluation

Filtered evaluation removes other known true answers from the candidate set while preserving the target entity being evaluated.

For validation queries, the filtering set is the union of:

- training triples;
- validation triples.

Test triples must not be consulted during validation filtering.

For test queries, the filtering set is the union of:

- training triples;
- validation triples;
- test triples.

For a tail query \((h,r,?)\), a candidate \(t'\neq t\) is removed if \((h,r,t')\) belongs to the permitted filtering set.

For a head query \((?,r,t)\), a candidate \(h'\neq h\) is removed if \((h',r,t)\) belongs to the permitted filtering set.

The target entity must always be restored if it appears in the filtering set.

This split-specific rule prevents test graph structure from influencing validation MRR or hyperparameter selection.

### Realistic rank

For each filtered query, the target rank is:

\(\operatorname{rank}
=
1+n_{>}
+\frac{1}{2}n_{=\mathrm{other}}\),

where:

- \(n_{>}\) is the number of unfiltered candidate entities with a score strictly greater than the target score;
- \(n_{=\mathrm{other}}\) is the number of other unfiltered candidate entities with a score exactly equal to the target score.

The target score must be obtained from the target entry in the scored candidate set. It must not be rounded or independently approximated.

Exact numerical equality is used for tie detection. No epsilon, tolerance, or approximate-equality function is permitted.

Near-equal floating-point scores are not ties.

The evaluation implementation must include reference tests covering:

- strict rankings;
- exact ties;
- near-ties;
- filtered true answers;
- restoration of the target;
- non-finite scores;
- equality between dense and chunked evaluation results.

### Ranking metrics

The official ranking metrics are:

- mean reciprocal rank, MRR;
- Hits@1;
- Hits@3;
- Hits@10.

MRR is computed as:

\(\operatorname{MRR}=\frac{1}{N}\sum_{q=1}^{N}\frac{1}{\operatorname{rank}_q}\).

Hits@\(k\) is the proportion of queries with realistic rank less than or equal to \(k\).

Every metric must be reported separately for:

- head-prediction queries;
- tail-prediction queries;
- the combined set of head- and tail-prediction queries.

The primary validation metric for hyperparameter selection is combined filtered validation MRR over all official head- and tail-prediction validation queries.

Test MRR, Hits, uncertainty metrics, or text-condition validation results must not be used to select training hyperparameters.

The primary ranking outcome for the final comparison is combined filtered test MRR over all official head- and tail-prediction test queries. Seen/unseen strata are mandatory secondary analyses and must not replace, redefine, or be pooled with this primary benchmark outcome.

### Seen and unseen reporting strata

An entity is training-seen if it occurs as a head or tail in at least one training triple.

For each prediction query, the target is classified as:

- seen-target;
- unseen-target.

Each source triple is also classified as:

- both-endpoints-seen;
- at-least-one-endpoint-unseen.

Ranking metrics must be reported for:

- all queries;
- seen-target queries;
- unseen-target queries;
- queries originating from both-endpoints-seen triples;
- queries originating from at-least-one-endpoint-unseen triples.

The candidate vocabulary and filtering rules remain unchanged across these strata.

Every stratum must report its query count. Empty strata must be reported as not applicable rather than silently omitted.

### Predictive probability distribution

KGE scores are not treated directly as probabilities.

For each filtered query \(q\), the candidate scores are converted to a probability distribution using temperature-scaled softmax:

\(p(e\mid q,T)
=
\frac{\exp(s_e/T)}
{\sum_{j\in C_q}\exp(s_j/T)}\),

where:

- \(C_q\) is the filtered candidate set;
- \(s_e\) is the score of candidate entity \(e\);
- \(T>0\) is the temperature.

The implementation must use a numerically stable log-sum-exp calculation.

The raw predictive distribution uses \(T=1\).

### Filtered-confidence interpretation boundary

The distribution above is conditional on the benchmark's filtered candidate set. It is a measure of confidence among the candidates retained for a specific filtered link-prediction query.

It must not be interpreted as:

- an open-world probability that a triple is true;
- a deployment probability over all possible real-world entities;
- a probability directly comparable across datasets with different candidate vocabularies or filtering structures.

Filtering uses known true triples according to the validation/test boundaries fixed above. Consequently, the resulting uncertainty metrics evaluate confidence quality inside the established filtered benchmark protocol, not open-world factual uncertainty.

### Validation-only temperature scaling

Every completed dataset–model–condition–replicate run receives its own post-hoc temperature.

A single global temperature is fitted jointly over all head and tail validation queries belonging to that run.

The same temperature is used for:

- head and tail test queries;
- seen and unseen reporting strata;
- every test query belonging to the run.

Temperature is selected by minimising mean multiclass negative log-likelihood on the validation queries.

The optimisation is restricted to \(10^{-3}\leq T\leq10^{3}\) and must use a deterministic bounded one-dimensional optimisation procedure.

The optimisation method, numerical tolerance, fitted temperature, convergence status, and boundary status must be recorded.

A non-finite objective or optimisation failure invalidates calibration for the run and must terminate the official run group.

A fitted value at an optimisation boundary must be retained and explicitly flagged. The permitted range must not be expanded after observing test results.

Temperature scaling must not change candidate rankings. The implementation must verify that raw and calibrated ranks are identical.

Test queries must never be used to fit, select, or modify the temperature.

Because each completed run receives its own validation-fitted temperature, calibrated metrics measure how well that run can be post-hoc calibrated using held-out validation data. They do not by themselves show that text initialisation produced intrinsically better raw score confidence. Raw and calibrated outcomes must therefore both be reported and distinguished in the interpretation.

### Top-label confidence and correctness

For every query, the predicted entity is the retained candidate in \(C_q\) with the highest score.

If multiple candidates have exactly the same maximum score, the candidate with the smallest canonical entity index is selected.

The number of top-score ties must be recorded.

Top-label confidence is:

\(c_q=\max_{e\in C_q}p(e\mid q,T)\).

Top-label correctness is:

\(y_q=1\) if the deterministically selected predicted entity equals the target entity, and \(y_q=0\) otherwise.

Predictive uncertainty may be represented as \(u_q=1-c_q\).

The deterministic top-label tie rule is used only for binary correctness metrics. Realistic rank remains the official ranking tie treatment.

### Multiclass negative log-likelihood

For target entity \(e_q^*\), query-level multiclass negative log-likelihood is:

\(\operatorname{NLL}_q=-\log p(e_q^*\mid q,T)\).

The reported NLL is the mean over queries.

NLL is reported for both:

- raw probabilities with \(T=1\);
- calibrated probabilities with validation-fitted \(T\).

Lower NLL is better.

### Expected calibration error

Top-label ECE uses 15 fixed equal-width confidence bins over \([0,1]\).

Bins are left-closed and right-open, except that the final bin includes confidence 1.

For non-empty bin \(B_m\):

\(\operatorname{acc}(B_m)=\frac{1}{|B_m|}\sum_{q\in B_m}y_q\),

\(\operatorname{conf}(B_m)=\frac{1}{|B_m|}\sum_{q\in B_m}c_q\).

ECE is:

\(\operatorname{ECE}
=
\sum_{m=1}^{15}
\frac{|B_m|}{N}
\left|
\operatorname{acc}(B_m)
-
\operatorname{conf}(B_m)
\right|\).

Empty bins contribute zero.

Every ECE result must be accompanied by a bin-level table containing:

- bin boundaries;
- query count;
- mean confidence;
- empirical accuracy;
- absolute calibration gap.

ECE is reported for raw and calibrated probabilities. Lower ECE is better.

ECE must not be interpreted without ranking accuracy and proper scoring metrics.

### Top-label Brier score

The top-label Brier score is:

\(\operatorname{Brier}
=
\frac{1}{N}\sum_{q=1}^{N}(c_q-y_q)^2\).

It is reported for raw and calibrated confidence. Lower values are better.

It must be labelled as top-label Brier score and must not be confused with the full multiclass Brier score.

### Confidence discrimination

The AUROC of top-label confidence against binary correctness measures whether correct predictions tend to receive higher confidence than incorrect predictions.

AUROC is reported for raw and calibrated confidence. Higher values are better.

If a reporting stratum contains only correct predictions or only incorrect predictions, AUROC is undefined and must be reported as not applicable.

An undefined AUROC must not be replaced by zero, one, or another numeric value.

### Risk–coverage evaluation

Queries are sorted by decreasing confidence. Exact confidence ties are broken by deterministic query identifier.

At coverage \(k/N\), selective risk is:

\(\operatorname{risk}(k)
=
1-\frac{1}{k}\sum_{i=1}^{k}y_{(i)}\),

where \(y_{(i)}\) follows descending confidence order.

Area under the risk–coverage curve is:

\(\operatorname{AURC}
=
\frac{1}{N}\sum_{k=1}^{N}\operatorname{risk}(k)\).

Lower AURC is better.

Oracle AURC is computed using the same correctness labels but ordering all correct predictions before all incorrect predictions.

Excess AURC is:

\(\operatorname{E\text{-}AURC}
=
\operatorname{AURC}
-
\operatorname{AURC}_{\mathrm{oracle}}\).

Lower E-AURC is better, with zero representing oracle confidence ordering for the observed accuracy.

AURC and E-AURC are reported for raw and calibrated confidence.

### Predictive entropy

Normalised predictive entropy may be reported as a descriptive diagnostic:

\(H_{\mathrm{norm}}(q)
=
-\frac{\sum_{e\in C_q}p_e\log p_e}
{\log |C_q|}\).

A query with only one remaining candidate is assigned normalised entropy zero and must be separately counted.

Entropy is reported only as a secondary descriptive measure.

Lower entropy alone does not establish better uncertainty estimation, calibration, discrimination, or link-prediction performance.

### Primary and secondary uncertainty outcomes

The primary probabilistic uncertainty outcome is calibrated test multiclass NLL.

The primary selective-prediction outcome is calibrated test E-AURC.

The following are pre-specified secondary outcomes:

- raw NLL;
- raw and calibrated ECE;
- raw and calibrated top-label Brier score;
- raw and calibrated AUROC;
- raw and calibrated AURC;
- raw E-AURC;
- raw and calibrated normalised entropy;
- fitted temperature.

No single uncertainty metric is sufficient to establish improvement.

All uncertainty outcomes must be interpreted jointly with MRR, Hits@1, Hits@3, Hits@10, and top-label accuracy.

Statistical claim rules are fixed in Decision 11. Directional hypotheses and smallest effects of interest must pass the final pre-test freeze gate in Decision 12.

### Excluded core uncertainty analyses

Top-K ECE variants such as ECE@10, ECE@50, and ECE@100 are not part of the core confirmatory evaluation.

They may be performed only as explicitly labelled exploratory sensitivity analyses after the core results have been preserved.

Exploratory analyses must not replace or redefine the pre-specified core outcomes.

### Per-query evaluation artifact

Every official evaluation must produce an immutable per-query table containing at least:

- dataset;
- model;
- initialisation condition;
- replicate identifier;
- split;
- query identifier;
- prediction direction;
- head, relation, and tail identifiers;
- target entity index;
- target seen status;
- triple endpoint-seen status;
- filtered candidate count;
- target score;
- realistic target rank;
- predicted entity index;
- maximum score;
- top-score tie count;
- top-label correctness;
- raw target probability;
- calibrated target probability;
- raw confidence;
- calibrated confidence;
- raw NLL;
- calibrated NLL;
- raw normalised entropy;
- calibrated normalised entropy;
- fitted temperature.

The table, aggregate metric report, calibration-bin table, and risk–coverage table must each receive a SHA-256 hash.

### Evaluation failure conditions

Evaluation must terminate if any of the following occurs:

- the target entity is absent from the candidate vocabulary;
- the target is not restored after filtering;
- the filtered candidate set is empty;
- a candidate score is non-finite;
- a probability, rank, confidence, NLL, or entropy value is non-finite;
- raw and calibrated candidate rankings differ;
- the number of generated queries differs from twice the number of evaluated triples;
- an aggregate metric cannot be reproduced from the preserved per-query artifact.

Test evaluation may be executed only after the protocol, data artifacts, selected hyperparameters, checkpoints, and calibration procedure have been frozen.

## Decision 10: Hyperparameter selection

Hyperparameters are selected separately for each dataset–model combination using only the Random initialisation condition and the validation split.

The six independent tuning units are:

- FB15k-237 with TransE;
- FB15k-237 with DistMult;
- WN18RR with TransE;
- WN18RR with DistMult;
- CoDEx-M with TransE;
- CoDEx-M with DistMult.

No correct-text or shuffled-text run may be trained or evaluated before the corresponding dataset–model hyperparameters have been frozen.

Test triples, test rankings, and test uncertainty metrics must not be accessed during hyperparameter selection.

### Selection objective

The sole hyperparameter-selection objective is combined filtered validation MRR as defined in Decision 9.

The following must not influence hyperparameter selection:

- validation uncertainty metrics;
- validation results from correct-text or shuffled-text conditions;
- test metrics;
- relation-specific results;
- seen/unseen subgroup results;
- legacy pilot results;
- visual preference for a loss curve;
- runtime, except for an exact metric tie.

Temperature scaling is not fitted or used during hyperparameter selection.

### Fixed training parameters

The following parameters are fixed and are not searched:

- entity embedding dimension: 384;
- relation embedding dimension: 384;
- optimiser: Adam;
- Adam \(\beta_1=0.9\);
- Adam \(\beta_2=0.999\);
- Adam \(\epsilon=10^{-8}\);
- optimiser weight decay: 0;
- pairwise logistic ranking objective from Decision 8;
- uniform negative-sampling procedure from Decision 8;
- embedding initialisation and normalisation rules from Decision 6;
- no condition-specific early stopping.

The Adam settings follow the documented PyTorch defaults except that optimiser weight decay is explicitly fixed to zero because regularisation is implemented through the batch-local term defined in Decision 8.

Source: [PyTorch Adam documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).

### Search space

For both TransE and DistMult, learning rate is sampled log-uniformly from:

\(10^{-4}\leq\operatorname{learning\ rate}\leq3\times10^{-3}\).

A learning-rate draw is generated by sampling uniformly in log space and exponentiating the result. The full-precision sampled value must be preserved in the configuration artifact.

Batch size is sampled uniformly from:

- 256;
- 512;
- 1024.

The number of negative triples per positive triple, \(K\), is sampled uniformly from:

- 1;
- 16;
- 64.

The batch-local L2 coefficient, \(\lambda\), is sampled uniformly from:

- 0;
- \(10^{-4}\);
- \(10^{-3}\);
- \(10^{-2}\);
- \(10^{-1}\).

For TransE, the scoring distance norm is selected from:

- L1, \(p=1\);
- L2, \(p=2\).

For DistMult, no distance-norm parameter exists.

The TransE L1/L2 alternatives follow the original TransE model definition.

Source: [Bordes et al., Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf).

The DistMult scoring function remains the standard bilinear function defined in Decision 4.

Source: [Yang et al., Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575).

### Search-space interpretation

The search space is defined before tuning and does not claim that any single framework default is universally optimal.

The learning-rate interval brackets the Adam default on a logarithmic scale.

The batch-size candidates are powers of two covering three levels of gradient aggregation.

The negative-sample candidates cover minimal, intermediate, and larger negative sets.

Because all embeddings have unit norm at \(t=0\), the regularisation candidates cover zero regularisation through a contribution that is substantial relative to the initial ranking-loss scale.

No candidate value is inherited solely from a legacy result.

### Hardware feasibility preflight

Before hyperparameter tuning, the largest planned configuration must pass a scheduled GPU memory and execution preflight:

- batch size 1024;
- \(K=64\);
- embedding dimension 384;
- both TransE and DistMult;
- each dataset shape.

The preflight may execute only a small fixed number of training steps and must be marked `official_result: false`.

It must not compute validation or test metrics.

If the largest configuration is infeasible, the search space must not be silently changed. A documented protocol amendment is required before tuning begins.

### Random configuration generation

For each dataset–model tuning unit, eight distinct candidate configurations are generated before any candidate is trained.

Candidate generation uses a dedicated HPO-design RNG stream and a recorded HPO-design seed.

This RNG stream must be independent of all training, initialisation, shuffling, batching, and negative-sampling RNG streams.

For TransE:

- exactly four candidates use \(p=1\);
- exactly four candidates use \(p=2\);
- all remaining hyperparameters are sampled from the defined search space.

For DistMult, all eight candidates are sampled from the applicable search space.

Duplicate configurations are rejected and resampled.

Candidates receive immutable deterministic configuration identifiers before training.

The complete eight-candidate design must be saved and hashed before the first tuning run begins.

Random search is used because it explores combinations more efficiently than an exhaustive grid when only some hyperparameters strongly affect validation performance.

Source: [Bergstra and Bengio, Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html).

### Tuning seeds

Three tuning seeds are used.

The tuning seeds must:

- be fixed before tuning begins;
- be identical across candidate configurations within a tuning unit;
- be distinct from the five official experimental seeds;
- use the independent RNG streams required by earlier decisions;
- be recorded in the frozen tuning configuration.

The exact numerical tuning and official seeds are fixed in Decision 11.

### Stage 1: Screening

All eight candidates are trained from \(t=0\) using the first tuning seed for exactly 25 epochs.

Every candidate is evaluated on the full validation split at epoch 25.

Candidates are ranked by combined filtered validation MRR.

The two valid candidates with the highest validation MRR advance to Stage 2.

An invalid or non-finite run receives no replacement seed and cannot advance.

If fewer than two candidates are valid, the tuning unit must terminate.

### Stage 2: Confirmation

The two advancing candidates are each trained from \(t=0\) using the first two tuning seeds for exactly 75 epochs.

Each candidate is evaluated on the full validation split at epoch 75.

For each candidate, the mean combined filtered validation MRR across the two tuning seeds is computed.

The candidate with the highest mean validation MRR is selected for Stage 3.

If either required seed trajectory is invalid, the candidate is invalid.

Both Stage 2 candidates must complete both required seed trajectories. Otherwise, the tuning unit must terminate.

### Stage 3: Final tuning

The single selected candidate is trained from \(t=0\) using all three tuning seeds for a maximum of 200 epochs.

The full validation split is evaluated every 20 epochs.

For each evaluation epoch, mean combined filtered validation MRR is computed across the three tuning seeds.

The hyperparameter configuration selected in Stage 2 is frozen. The official epoch budget is the evaluated epoch with the highest mean combined filtered validation MRR in Stage 3.

Stage 3 trajectories must be trained independently from \(t=0\); Stage 1 or Stage 2 optimiser state must not be reused.

All three Stage 3 seed trajectories must complete successfully. Otherwise, the tuning unit must terminate.

The 200-epoch maximum is a pre-declared resource ceiling, not an assertion that 200 epochs are universally optimal.

If the selected epoch is 200, the result must be flagged as right-censored at the tuning horizon. The horizon must not be extended after observing text-condition or test results.

### Deterministic tie-breaking

Stage rankings use full-precision stored validation MRR values.

For an exact Stage 1 tie, candidates are ordered by:

1. smaller \(K\);
2. smaller batch size;
3. lexicographically smaller configuration identifier.

For an exact Stage 2 tie, the same rule is used.

For an exact Stage 3 tie between epochs, the earlier epoch is selected.

No approximate tie tolerance is used.

### Invalid-run rules

A tuning trajectory is invalid if it contains:

- non-finite loss;
- non-finite embedding values;
- non-finite validation scores or ranks;
- a failed paired RNG or data-integrity assertion;
- an unexpected process termination;
- a checkpoint or configuration hash mismatch.

An invalid run must remain in the tuning record.

It must not be silently restarted with a different seed, configuration, learning rate, or software environment.

An infrastructure failure that occurs before model execution may be retried only with the identical configuration, seed, environment, and scheduler request, with both attempts linked in the audit record.

### Configuration selection and freezing

Every dataset–model tuning unit produces one frozen configuration containing at least:

- learning rate;
- batch size;
- \(K\);
- \(\lambda\);
- TransE distance norm where applicable;
- official epoch budget;
- optimiser settings;
- model dimension;
- tuning seeds;
- search-design seed;
- source commit;
- environment hash;
- data-manifest hashes.

The frozen configuration applies unchanged to:

- random initialisation;
- correct-text initialisation;
- shuffled-text initialisation;
- all five official experimental seeds.

Official runs do not use condition-specific early stopping or condition-specific hyperparameter changes.

### Separation of tuning and official results

Tuning runs are development artifacts and are not part of the 90-run official result matrix.

The official matrix remains:

\(3\text{ datasets}\times2\text{ models}\times3\text{ conditions}\times5\text{ seeds}=90\text{ runs}\).

Tuning seeds must not be included in official means, confidence intervals, hypothesis tests, or result tables.

Official random-baseline runs must be retrained from \(t=0\) using the five official seeds after hyperparameters have been frozen.

Because tuning is performed only on the Random condition, the official contrasts estimate the effect of initialisation under a shared configuration selected for the Random baseline. They do not estimate the best independently attainable performance of each initialisation condition.

### Pre-declared trajectory budget

For each of the six dataset–model tuning units, the tuning design contains:

- eight Stage 1 trajectories;
- four Stage 2 trajectories, formed by two candidates and two tuning seeds;
- three Stage 3 trajectories, formed by one candidate and three tuning seeds.

This yields 15 tuning trajectories per unit and 90 tuning trajectories in total. Together with the 90 official runs, the planned experiment contains 180 training trajectories, excluding non-result-producing hardware and toy-graph preflights.

### Search-space amendments

The search space must not be expanded merely because:

- the best candidate lies near a boundary;
- validation MRR is lower than expected;
- a legacy result was better;
- text initialisation is expected to prefer a different setting;
- a test result is disappointing.

If a tuning unit cannot produce a valid configuration, work must stop before official runs.

Any amendment must:

1. explain the concrete failure;
2. be written before rerunning tuning;
3. create a new protocol commit;
4. invalidate and restart the affected tuning unit;
5. remain independent of correct-text, shuffled-text, and test results.

### Tuning artifacts

Each tuning unit must preserve and hash:

- the complete search-space specification;
- the eight generated candidate configurations;
- HPO-design seed;
- tuning seeds;
- stage membership;
- all run configurations;
- training-loss histories;
- validation-MRR histories;
- checkpoint hashes;
- invalid-run records;
- Stage 1 ranking;
- Stage 2 ranking;
- Stage 3 epoch-selection table;
- final selection report;
- frozen official configuration.

The final selection must be reproducible mechanically from the preserved artifacts without manual interpretation.

### Reproducibility boundary

Tuning and official training must use the same pinned software environment and numerical precision.

PyTorch deterministic-algorithm settings and all RNG seeds must be recorded.

Determinism is required within the frozen software and hardware environment; exact reproducibility across different PyTorch releases or hardware platforms is not assumed.

Source: [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Decision 11: Random seeds and multi-seed statistical analysis

The official experimental unit is a complete paired training replicate, not an individual test query.

For each dataset–model–seed combination, random, correct-text, and shuffled-text runs form one paired block.

All seed values and derivation rules are fixed before hyperparameter tuning and official training.

### Seed derivation rule

Seeds are derived from SHA-256 rather than manually selected.

The fixed UTF-8 prefix is:

`Integrating External Evidence to Reduce Uncertainty in Link Prediction`

For replicate seeds, the input string is:

`prefix + "|" + group + "|" + index`

where `group` is either `tuning` or `official`, and `index` begins at 1.

The SHA-256 digest is computed over the exact UTF-8 string.

The first four digest bytes are interpreted as one unsigned 32-bit big-endian integer.

The input string, complete SHA-256 digest, and derived integer must be recorded.

### Tuning seeds

The three tuning seeds are:

| Tuning replicate | Seed |
|---:|---:|
| 1 | 347869531 |
| 2 | 2594257857 |
| 3 | 3877759316 |

Tuning seeds may be used only for Random-baseline hyperparameter selection.

They must not appear in the 90-run official result matrix.

### Official seeds

The five official seeds are:

| Official replicate | Seed |
|---:|---:|
| 1 | 3239737201 |
| 2 | 3693500706 |
| 3 | 4270284857 |
| 4 | 3089462143 |
| 5 | 3051266068 |

All dataset–model–condition official runs use these five replicate identifiers.

The same five seeds must be retained regardless of observed performance.

No seed may be removed or replaced because its result is unusually high, low, unstable, or difficult to interpret.

### HPO-design seeds

HPO candidate generation uses the following independently derived seeds:

| Dataset | Model | HPO-design seed |
|---|---|---:|
| FB15k-237 | TransE | 3687661691 |
| FB15k-237 | DistMult | 1578249420 |
| WN18RR | TransE | 61651425 |
| WN18RR | DistMult | 3751609374 |
| CoDEx-M | TransE | 1196443311 |
| CoDEx-M | DistMult | 2640021856 |

For HPO design, the SHA-256 input string is:

`prefix + "|hpo-design|" + dataset + "|" + model`

These seeds are used only to generate the eight candidate configurations required by Decision 10.

### Independent RNG namespaces

A base replicate seed must not be reused as one shared mutable RNG stream.

Independent substream seeds are derived with SHA-256 using explicit namespaces.

For model-specific streams, the derivation input is:

`prefix + "|stream|" + group + "|" + replicate_index + "|" + dataset + "|" + model + "|" + namespace`

The model-specific namespaces are:

- `random_entity_initialisation`;
- `relation_initialisation`;
- `batch_order`;
- `negative_sampling`;
- `data_loader_workers`.

For the shuffled-text permutation, which is shared across TransE and DistMult for the same dataset replicate, the model name is omitted:

`prefix + "|stream|" + group + "|" + replicate_index + "|" + dataset + "|shuffle"`

Every substream seed is obtained from the first four SHA-256 digest bytes as an unsigned 32-bit big-endian integer.

The complete derivation table must be generated and hashed before the associated runs begin.

Within one dataset–model–official-seed paired block:

- relation-initialisation seeds are identical across all three conditions;
- batch-order seeds are identical across all three conditions;
- negative-sampling seeds are identical across all three conditions;
- data-loader-worker seeds are identical across all three conditions;
- the random-entity-initialisation stream is used only by the Random condition;
- the shuffle stream is used only to create the precomputed shuffled-text artifact.

Using separate RNG namespaces prevents a change in one random process from shifting another random sequence.

Source: [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html).

### Official paired blocks

For each dataset, model, and official seed, the complete paired block contains:

- one Random run;
- one Correct-text run;
- one Shuffled-text run.

The official design therefore contains:

\(3\text{ datasets}\times2\text{ models}\times5\text{ seeds}=30\text{ paired blocks}\),

and:

\(30\text{ paired blocks}\times3\text{ conditions}=90\text{ official runs}\).

Paired conditions must use the matched stochastic components required by Decisions 6, 7, and 8.

### Statistical unit

The training replicate seed is the unit of statistical comparison.

Test queries are nested observations used to compute one metric value for one trained run.

Thousands of head- and tail-prediction queries must not be treated as thousands of independent training replicates.

For every dataset–model combination and metric, condition comparisons are performed using five seed-level paired differences.

A query-level bootstrap may be performed only as a clearly labelled exploratory sensitivity analysis.

It must not replace the official training-seed analysis.

### Pre-specified condition contrasts

The primary condition contrasts are:

1. Correct-text versus Random;
2. Correct-text versus Shuffled-text.

Correct-text versus Random estimates the total effect of introducing correctly aligned text-derived initialisation.

Correct-text versus Shuffled-text estimates the effect of correct semantic entity–text alignment.

Shuffled-text versus Random is a pre-specified secondary mechanistic contrast.

It estimates effects associated with pretrained text-vector geometry without correct semantic alignment.

### Improvement-oriented paired differences

For metrics where higher values are better, the paired difference is:

\(d_s=M_{\mathrm{condition},s}-M_{\mathrm{control},s}\).

Higher-is-better metrics include:

- MRR;
- Hits@1;
- Hits@3;
- Hits@10;
- AUROC.

For metrics where lower values are better, the paired difference is:

\(d_s=M_{\mathrm{control},s}-M_{\mathrm{condition},s}\).

Lower-is-better metrics include:

- NLL;
- ECE;
- top-label Brier score;
- AURC;
- E-AURC.

Therefore, \(d_s>0\) always represents improvement by the first-named condition.

Entropy and fitted temperature do not receive an automatic improvement direction. They remain descriptive.

All result tables must also preserve the original untransformed condition means so that the sign convention is transparent.

### Seed-level summaries

For every dataset–model–metric–contrast combination, report:

- all five paired differences;
- arithmetic mean;
- median;
- sample standard deviation using \(n-1\) degrees of freedom;
- minimum;
- maximum;
- number of positive differences;
- number of zero differences;
- number of negative differences;
- two-sided 95% confidence interval for the mean paired difference.

No seed-level value may be hidden from the result artifact or replaced by only an aggregate statistic.

### Confidence interval

For five complete paired differences, the confidence interval is:

\(\bar d
\pm
t_{0.975,4}\frac{s_d}{\sqrt{5}}\),

where:

- \(\bar d\) is the mean paired difference;
- \(s_d\) is the sample standard deviation;
- \(t_{0.975,4}\) is the 97.5th percentile of the Student t distribution with four degrees of freedom.

The interval is an unadjusted seed-level 95% confidence interval and must be labelled as such.

Because \(n=5\) is small, confidence intervals may be wide and normality cannot be assessed reliably.

An interval containing zero must not be interpreted as proof of no effect.

Source: [Colas et al., How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments](https://inria.hal.science/hal-01890154/file/1806.08295.pdf).

### Paired hypothesis tests

For primary outcomes, a two-sided paired t-test is computed from the five paired differences.

Two-sided tests are retained even if later hypotheses are directional because adverse effects remain scientifically relevant.

A raw p-value and a multiplicity-adjusted p-value must both be reported.

P-values are supportive summaries and must not replace effect sizes, seed-level values, or confidence intervals.

If the paired-difference standard deviation is zero:

- if every difference is zero, the raw p-value is reported as 1;
- if the common difference is non-zero, the t-test is undefined and its p-value is reported as not applicable.

Undefined values must not be replaced with zero.

### Confirmatory metric families

Six confirmatory families are defined.

Each family contains the six dataset–model tests formed by three datasets and two models.

The families are:

1. Correct-text versus Random for MRR;
2. Correct-text versus Shuffled-text for MRR;
3. Correct-text versus Random for calibrated NLL;
4. Correct-text versus Shuffled-text for calibrated NLL;
5. Correct-text versus Random for calibrated E-AURC;
6. Correct-text versus Shuffled-text for calibrated E-AURC.

Within each family, raw two-sided paired t-test p-values are adjusted using the Holm sequentially rejective procedure with family-wise \(\alpha=0.05\).

No p-value may be transferred between families.

Source: [Holm, A Simple Sequentially Rejective Multiple Test Procedure](https://www.jstor.org/stable/4615733).

### Secondary outcomes

The following are pre-specified secondary outcomes:

- Hits@1;
- Hits@3;
- Hits@10;
- raw NLL;
- raw and calibrated ECE;
- raw and calibrated top-label Brier score;
- raw and calibrated AUROC;
- raw and calibrated AURC;
- raw E-AURC;
- raw and calibrated normalised entropy;
- fitted temperature;
- Shuffled-text versus Random contrasts;
- seen/unseen subgroup outcomes.

Secondary outcomes receive the same paired-difference summaries and confidence intervals.

They do not receive additional confirmatory significance claims.

Any additional metric or subgroup introduced after official evaluation begins must be labelled exploratory.

### Cross-dataset reporting

Dataset–model units remain separate.

Queries from different datasets must not be pooled.

The six dataset–model effects must not be averaged into one confirmatory grand mean because the datasets differ in graph structure, domain, text provenance, relation composition, and evaluation difficulty.

A forest plot may display the six effects and confidence intervals.

The number of dataset–model units with positive, zero, or negative effects may be reported descriptively.

Such counts are not additional hypothesis tests.

### Uncertainty interpretation

Calibrated NLL is the primary probabilistic uncertainty outcome.

Calibrated E-AURC is the primary selective-prediction uncertainty outcome.

For a given contrast:

- `directionally consistent evidence` means that the mean paired differences for both primary uncertainty outcomes favour the first-named condition;
- `partial evidence` means one primary uncertainty outcome improves while the other remains directionally unclear;
- `mixed evidence` means the two primary uncertainty outcomes move in opposite directions;
- `no primary evidence` means only secondary metrics such as ECE improve.

The phrase `supported uncertainty improvement` is reserved for a result that satisfies all pre-frozen confirmatory and practical criteria: both primary uncertainty outcomes must favour the first-named condition, satisfy their pre-specified smallest effects of interest, and pass the corresponding Holm-adjusted confirmatory tests; ranking deterioration must also remain within the pre-specified MRR guardrail. If suitable smallest effects of interest cannot be justified before official evaluation, this stronger phrase must not be used.

A lower ECE alone is insufficient to claim improved uncertainty estimation.

Every uncertainty claim must be presented jointly with:

- MRR;
- Hits@1;
- calibrated NLL;
- calibrated E-AURC;
- raw and calibrated ECE.

If uncertainty outcomes improve while ranking performance deteriorates beyond a pre-specified smallest effect size of interest, the result must be described as an accuracy–uncertainty trade-off rather than an overall improvement.

### Practical significance boundary

Absolute paired differences must be reported in the natural metric scale.

Relative percentage changes may be reported only when the control metric is non-zero and must not replace absolute changes.

Smallest effects of interest are not defined from legacy pilot results or official test outcomes.

They must be justified through the literature review and locked together with the directional hypotheses before official test evaluation.

Until those thresholds are locked, the protocol must not use phrases such as:

- practically equivalent;
- no meaningful deterioration;
- practically important improvement;
- non-inferior.

### Missing and failed official runs

A confirmatory dataset–model analysis requires five complete paired blocks.

If one condition in a paired block fails because of:

- non-finite model values;
- integrity-check failure;
- reproducible model-training failure;
- missing or invalid artifacts;

then:

- the seed must not be replaced;
- the failed condition must not be silently excluded;
- the other two conditions must not be analysed as an unpaired substitute;
- the entire paired block is incomplete;
- the dataset–model confirmatory analysis must stop if fewer than five complete paired blocks remain.

Incomplete blocks and their causes remain reportable outcomes.

A scheduler, node, filesystem, or hardware failure that occurs independently of model state may be retried only with the identical:

- code commit;
- configuration;
- dataset and artifact hashes;
- seed;
- scheduler resources;
- software environment.

The original and retry job identifiers must both be preserved.

### No selective reruns

An official run must not be repeated merely because:

- its metric is unusually low or high;
- its loss curve differs from another seed;
- it weakens a hypothesis;
- it produces an inconvenient trade-off;
- it changes a p-value or confidence interval.

A run may be invalidated only by a pre-specified integrity failure or documented infrastructure failure.

### Statistical artifacts

The analysis pipeline must preserve and hash:

- the complete seed-derivation table;
- base labels and SHA-256 digests;
- RNG namespace table;
- per-run metric table;
- per-seed paired-difference table;
- summary-statistics table;
- confidence-interval table;
- raw p-value table;
- Holm-adjusted p-value table;
- family definitions;
- failure and retry records;
- analysis configuration;
- analysis code commit;
- generated result tables and figures.

Every aggregate result must be mechanically reproducible from preserved per-run metrics.

Manual copying of headline results into the official results table is not permitted.

## Decision 12: Final freeze gates, numerical execution, and cluster policy

Approval of the method protocol permits implementation, source auditing, unit tests, scheduled preflights, and Random-only hyperparameter tuning after the applicable gates below have passed. It does not by itself permit official result-producing runs.

### Gate A: Input and source freeze

Before text encoding or model tuning, every dataset and evidence source must have a machine-readable manifest recording at least:

- canonical source URL and source project;
- exact repository commit, release, or version;
- artifact path and filename;
- byte size and SHA-256 hash;
- retrieval date;
- parsing and vocabulary-alignment rule;
- split sizes, entity counts, relation counts, and duplicate/overlap checks;
- complete-vocabulary coverage and evidence-status counts.

The FB15k-237 and CoDEx-M evidence files must be identified by exact source revision and file hash. The WN18RR manifest must identify the exact entity-aligned part-of-speech mapping artifact and prove its compatibility with Princeton WordNet 3.0. Trying multiple parts of speech and accepting the first match is prohibited.

The Hugging Face revision and all encoder settings required by Decision 5 must also be frozen. Official encoding and training jobs must consume only the pinned local artifacts; they must not depend on mutable live API responses.

Failure of any coverage, uniqueness, version, alignment, or hash check terminates the affected pipeline before model training.

### Gate B: Numerical execution freeze

The official numerical policy is:

- model parameters, training batches, optimiser state, and score generation use IEEE 754 float32;
- automatic mixed precision, autocast, and gradient scaling are disabled;
- CUDA TF32 matrix multiplication and cuDNN TF32 are disabled;
- PyTorch float32 matrix-multiplication precision is set to `highest`;
- cuDNN benchmarking is disabled;
- deterministic PyTorch algorithms are enabled;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA initialisation;
- ranking comparisons use the preserved float32 candidate scores and the exact-tie rule in Decision 9;
- log-sum-exp, temperature fitting, probability normalisation, and aggregate metric reductions use float64 calculations derived from those preserved scores.

The run manifest must record the Python, PyTorch, CUDA, cuDNN, driver, package, GPU, host, environment, and determinism settings. A required operation that cannot run under the deterministic policy must fail during preflight and requires a documented protocol amendment before tuning.

Exact bitwise reproduction across different hardware or software releases is not claimed. Reproduction is required within the frozen execution environment.

### Gate C: Cluster execution policy

All cluster-side data downloading, archive extraction, description generation, text encoding, testing that traverses research data, training, calibration, and evaluation must execute through the scheduler on an allocated compute node.

The login node is limited to lightweight navigation, version-control inspection, file transfer, job submission, queue inspection, and viewing small logs. Research Python scripts, filesystem-intensive processing, model inference, training, and autonomous coding agents must not run on the login node.

Every scheduled job must record at least:

- scheduler job identifier;
- partition, node, requested CPUs, memory, and GPU resources;
- hostname and visible CUDA devices;
- start and completion time;
- job-local temporary directory;
- code commit and working-tree status;
- configuration and input-manifest hashes;
- exit status and output-artifact hashes.

Infrastructure retries must follow the identical-retry rule in Decisions 10 and 11.

### Gate D: Implementation acceptance

Before HPO, automated tests must cover at least:

- hand-calculated TransE and DistMult scores;
- filtered head and tail ranking;
- exact realistic-rank tie handling;
- agreement between dense and chunked evaluation;
- rejection of non-finite inputs;
- deterministic and independent RNG streams;
- negative-sampling rejection of training-positive triples;
- shuffled-text bijection, derangement, hash, and cross-model sharing rules;
- preservation of training-unseen entity rows;
- equality of paired relation initialisation and training batches;
- equality of Correct-text and Shuffled-text vector multisets;
- invariance of rankings under temperature scaling;
- reproduction of aggregate metrics from per-query artifacts.

A scheduled toy-graph GPU run and the largest-configuration memory preflight from Decision 10 must pass with `official_result: false`. Test reports, preflight reports, code commit, and environment manifest must be preserved and hashed.

### Gate E: Promotion to official runs

The experiment may progress only in this order:

1. freeze dataset, evidence, encoder, and environment manifests;
2. pass unit, integration, toy-graph, scheduler, and hardware preflights;
3. generate and hash the HPO designs;
4. complete Random-only HPO and freeze one configuration and epoch budget for every dataset–model unit;
5. freeze directional hypotheses, smallest effects of interest, and the MRR accuracy guardrail without consulting Correct-text, Shuffled-text, or test results;
6. generate and hash the complete 90-run official matrix;
7. execute the paired official runs;
8. fit temperatures on validation data and evaluate the already-frozen checkpoints on test data;
9. generate statistical tables mechanically from the preserved run artifacts.

Passing a gate must create an immutable report. A later failure cannot be bypassed by manual editing, silent fallback, seed replacement, or selective rerunning.

## Approval record

- Decision 1 approved by the protocol owner on 2026-08-04.
- Decision 2 approved by the protocol owner on 2026-08-04.
- Research questions approved by the protocol owner on 2026-08-04.
- Decision 3 approved by the protocol owner on 2026-08-04.
- Decision 4 approved by the protocol owner on 2026-08-04.
- Decision 5 approved by the protocol owner on 2026-08-04.
- Decision 6 approved by the protocol owner on 2026-08-04.
- Decision 7 approved by the protocol owner on 2026-08-04.
- Decision 8 approved by the protocol owner on 2026-08-04.
- Decision 9 approved by the protocol owner on 2026-08-04.
- Decision 10 approved by the protocol owner on 2026-08-04.
- Decision 11 approved by the protocol owner on 2026-08-04.
- Decision 12 and the final audit amendments approved by the protocol owner on 2026-08-04.


## Pending decisions

- Directional hypotheses, metric-specific smallest effects of interest, and the MRR accuracy guardrail following the literature review.
- Exact dataset, evidence-source, WordNet part-of-speech mapping, and encoder manifests required by Gate A.
- Frozen software, hardware, and numerical environment manifest required by Gate B.
- Successful implementation and scheduled preflight reports required by Gate D.
- Six selected hyperparameter configurations and official epoch budgets produced by the pre-registered HPO procedure.
