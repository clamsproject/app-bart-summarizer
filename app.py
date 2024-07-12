"""
DELETE THIS MODULE STRING AND REPLACE IT WITH A DESCRIPTION OF YOUR APP.

app.py Template

The app.py script does several things:
- import the necessary code
- create a subclass of ClamsApp that defines the metadata and provides a method to run the wrapped NLP tool
- provide a way to run the code as a RESTful Flask service


"""

import argparse
import logging
import warnings

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

from summarizer import TextSummarizer
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

# For an NLP tool we need to import the LAPPS vocabulary items


class BartSummarizer(ClamsApp):

    def __init__(self):
        self.summarizer = TextSummarizer()
        super().__init__()

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory.
        # When using the ``metadata.py`` leave this do-nothing "pass" method here.
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._annotate

        # Register a new view of the bart-summarizer app
        new_view = mmif.new_view()
        self.sign_view(new_view, parameters)

        views_contain_doc = mmif.get_all_views_contain(DocumentTypes.TextDocument)
        if views_contain_doc:
            view_to_summarize = views_contain_doc[-1]
            for doc in view_to_summarize.get_documents():
                self._run_bart(new_view, doc)
        else:
            other_docs = mmif.get_documents_by_type(DocumentTypes.TextDocument)
            if other_docs:
                for doc in other_docs:
                    self._run_bart(new_view, doc)
            else:
                warnings.warn("No text documents found in the input MMIF. No summarization performed.")

        return mmif

    def _run_bart(self, view: View, doc: str) -> None:
        summarized_text = view.new_textdocument(self.summarizer.summarize_text(doc.text_value))
        new_alignment = view.new_annotation(at_type=AnnotationTypes.Alignment,
                                           properties={'source': doc.long_id, 'target': summarized_text.long_id})

def get_app():
    """
    This function effectively creates an instance of the app class, without any arguments passed in, meaning, any
    external information such as initial app configuration should be set without using function arguments. The easiest
    way to do this is to set global variables before calling this.
    """
    # for example:
    # return TextSummarizer(create, from, global, params)
    return BartSummarizer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    # if get_app() call requires any "configurations", they should be set now as global variables
    # and referenced in the get_app() function. NOTE THAT you should not change the signature of get_app()
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
