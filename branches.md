This is a short summary of the active branches in my fork of https://github.com/usnistgov/STVM_NLP_Research.

 - `master`/`main` - Maintenance, miscellaneous commits, or changes merged from other branches; sometimes a new branch will be created specifically for a pull request and sometimes (i.e., when a branch contains the exact commits wanted in the PR) the merge will be done with the original branch into the and fetched from the upstream repository
 - `testing` - Parent branch for tests, experiments, and other content that does not apply to the original paper and thus will probably not be included in a pull request
     - `alternate-encodings` - Experiments with adding new text encodings and models to improve the versatility of the combined system
 - `performance` - Adjustments and refactoring related to optimization using TensorFlow's XLA JIT compiler, Numba, or similar tools