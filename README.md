# SimDocSin

SimDocSin is a cross-lingual document similarity checking tool for Sinhala and English.

This system can be used to find similar documents or parts of documents of Sinhala (English) language to a given document of English (Sinhala) language. System consists of two parts.
##### Full Matching

Here an user can submit a source document to get any matching complete document that exists in the system database in target language. Here the user has to set 3 input fields. <br> 
* Input language - Language of the source document. It can be Sinhala or English.<br>
<br>
* Similarity level - This indicates how much similarity you expect from a similar pair output by system. The user can give a value within the range from 1 to 5. Low value means there is a high chance of getting a similar document but the similarity can be relatively low while high value means that system can output a document with a relatively high similarity but the chance of getting a result is low. <br>
<br>
* Source file - Source document given as the input. It can be submitted as either a file or a text.<br>
<br>
##### Partial Matching

Here an user can submit a source document to get any matching partials of documents that exist in the system database in target language. Here also the user has to set the 3 inputs fields mentioned in the previous section. Apart from those there is another input field called Min Length. <br> 
* Min Length - minimum number of sentences the user expected to have in a document partial. It can be 1, greater than 1, greater than 2, greater than 5 or greater than 10.<br>

## How to Deploy SimDocSin

### Install Dependencies
You will need python 3.x. You can build the enviornment as follows.<br>
```pip install -r requirements.txt```<br>
Run below command to install the LASER models needed for embeddings.<br>
```python -m laserembeddings download-models```<br>
### Create Data Container Folders
You have to create the following folders within SimDocSin directory.<br>
```db``` - To contain embedded files <br>
```index``` - To contain index files <br>
```inputs``` - To contain documents inputted by users to the system <br>
```outputs_full_match``` - To contain documents outputted by the system <br>
```outputs_partial_match``` - To contain documents outputted by the system <br>
### Embed Documents
Run below command to embed json list of documents using ```embedding_creator.py``` inside ```embedder``` folder.<br><br>
```python embedding_creator.py path/to/input_file.json path/to/output_file.json```

The format of the input json file should be in following format.<br>
For english documents
```
[
  {"content_en": "english document content"},
  ...
]
```
For sinhala documents
```
[
  {"content_si": "sinhala document content"},
  ...
]
```
For parallel documents
```
[
  {
   "content_en": "english document content",
   "content_si": "sinhala document content"
  },
  ...
]
```
### Build Document Database
Update paths to embed json files inside ```filename.py```

Run ```db_split.py``` to build the document database.<br><br>
For english<br>
```python db_split.py en```<br>
For sinhala<br>
```python db_split.py si```

### Build Index Files
Run ```indexing.py``` to build index files.<br>(You should use same filename.py for both previous step and this step )<br><br>
For english<br>
```python indexing.py en```<br>
For sinhala<br>
```python indexing.py si```

### Run SimDocSin
```flask run```

### Contributors
Udhan Isuranga (udhanisuranga.16@cse.mrt.ac.lk) <br>
Janaka Sandaruwan (janakasadaruwan.16@cse.mrt.ac.lk) <br>
Udesh Athukorala (udeshathukorala.16@cse.mrt.ac.lk) <br>
