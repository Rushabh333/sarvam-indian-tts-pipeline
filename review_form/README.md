# Manual Review Google Form Pack

This folder contains the 37-clip manual review setup.

Files:
- `selected_37_review_manifest.csv`: reviewer/debug manifest with public audio URLs.
- `selected_37_review_manifest.json`: same manifest in JSON.
- `create_google_form.gs`: paste into Google Apps Script and run `createSarvamReviewForm()`.
- `public_upload/review_form_37/`: audio files uploaded to the public HuggingFace dataset.

The Google Form uses public HuggingFace audio links because this environment does not have Google account OAuth configured. Once the Apps Script runs from your Google account, it creates the actual Google Form and prints both the edit URL and response URL in the Apps Script logs.
