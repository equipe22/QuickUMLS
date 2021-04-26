import spacy
from spacy.tokens import Span
from spacy.strings import StringStore
from spacy.language import Language
from spacy.lang.fr import French

from .core import QuickUMLS
from . import constants


@French.factory("quickumls", default_config={
            'quickumls_fp' : 'umls_fr', 
            'best_match': True, 
            'ignore_syntax': False, 
            'overlapping_criteria':'score',
            'threshold':0.9, 
            'similarity_name':'jaccard', 
            'min_match_length':3,
            'accepted_semtypes': constants.ACCEPTED_SEMTYPES,
            'verbose':True, 
            'keep_uppercase':False})
def create_quickumls(nlp: Language, name: str, 
                     quickumls_fp, 
                     best_match, 
                     ignore_syntax,
                     overlapping_criteria,
                     threshold,
                     similarity_name, 
                     min_match_length,
                     accepted_semtypes,
                     verbose, 
                     keep_uppercase):

    return SpacyQuickUMLS(nlp, 
                          quickumls_fp, 
                          best_match=best_match, 
                          ignore_syntax = ignore_syntax,
                         overlapping_criteria = overlapping_criteria,
                         threshold = threshold,
                        similarity_name = similarity_name, 
                     min_match_length= min_match_length,
                     accepted_semtypes= accepted_semtypes,
                     verbose = verbose, 
                     keep_uppercase = keep_uppercase)



class SpacyQuickUMLS(object):
    name = 'QuickUMLS matcher'
    
    def __init__(self, nlp, quickumls_fp, best_match=True, ignore_syntax=False, **kwargs):
        """Instantiate SpacyQuickUMLS object

            This creates a QuickUMLS spaCy component which can be used in modular pipelines.  
            This module adds entity Spans to the document where the entity label is the UMLS CUI and the Span's "underscore" object is extended to contains "similarity" and "semtypes" for matched concepts.

        Args:
            nlp: Existing spaCy pipeline.  This is needed to update the vocabulary with UMLS CUI values
            quickumls_fp (str): Path to QuickUMLS data
            best_match (bool, optional): Whether to return only the top match or all overlapping candidates. Defaults to True.
            ignore_syntax (bool, optional): Wether to use the heuristcs introduced in the paper (Soldaini and Goharian, 2016). TODO: clarify,. Defaults to False
            **kwargs: QuickUMLS keyword arguments (see QuickUMLS in core.py)
        """
        
        self.quickumls = QuickUMLS(quickumls_fp, 
            # By default, the QuickUMLS objects creates its own internal spacy pipeline but this is not needed
            # when we're using it as a component in a pipeline
            spacy_component = True,
            **kwargs)
        
        # save this off so that we can get vocab values of labels later
        self.nlp = nlp
        
        # keep these for matching
        self.best_match = best_match
        self.ignore_syntax = ignore_syntax

        # let's extend this with some proprties that we want
        Span.set_extension('similarity', default = -1.0)
        Span.set_extension('semtypes', default = -1.0)
        
    def __call__(self, doc):
        # pass in the document which has been parsed to this point in the pipeline for ngrams and matches
        matches = self.quickumls._match(doc, best_match=self.best_match, ignore_syntax=self.ignore_syntax)
        
        # Convert QuickUMLS match objects into Spans
        for match in matches:
            # each match may match multiple ngrams
            for ngram_match_dict in match:
                start_char_idx = int(ngram_match_dict['start'])
                end_char_idx = int(ngram_match_dict['end'])
                
                cui = ngram_match_dict['cui']
                # add the string to the spacy vocab
                self.nlp.vocab.strings.add(cui)
                # pull out the value
                cui_label_value = self.nlp.vocab.strings[cui]
                
                # char_span() creates a Span from these character indices
                # UMLS CUI should work well as the label here
                span = doc.char_span(start_char_idx, end_char_idx, label = cui_label_value)
                # add some custom metadata to the spans
                span._.similarity = ngram_match_dict['similarity']
                span._.semtypes = ngram_match_dict['semtypes']
                                
                try:
                    doc.ents = list(doc.ents) + [span]
                except ValueError as e:
                    print(e, span, span.start, span.end)
                
                
        return doc