"""SNF Sylvester normalizing flows"""

"""Began with a transformed a version of the pytorch based SNF in https://github.com/riannevdberg/sylvester-flows to tensorflow"""


"""Used their first case: Orthogonal SNFs with R and R^\tilde being diagonal"""


"""Similar to the original version of SNFs, all the flow parameters (mainly R, R^\tilde and Q of each transformation) are fully amortised since they are produced as the output of the inference network."""


"""Constructed the tib generative and recogntion models accordingly on top of that."""
