# SimDocSin

SimDocSin is a cross-lingual document similarity checking tool for Sinhala and English.

This system can be used to find similar documents or parts of documents of Sinhala (English) language to a given document of English (Sinhala) language. System consists of two parts.
##### Full Matching

Here an user can submit a source document to get any matching complete document that exists in the system database in target language. Here the user has to set 3 input fields. <br> 
* Input language - Language of the source document. It can be Sinhala or English.
* Similarity level - This indicates how much similarity you expect from a similar pair output by system. The user can give a value within the range from 1 to 5. Low value means there is a high chance of getting a similar document but the similarity can be relatively low while high value means that system can output a document with a relatively high similarity but the chance of getting a result is low. <br>
* Source file - Source document given as the input. It can be submitted as either a file or a text.<br>

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
Run ```build.sh``` or ```build.bat```.  It will create the following folders within SimDocSin directory.<br>
```db``` - To contain embedded files preprocessed for indexing <br>
```index``` - To contain index files <br>
```inputs``` - To contain documents inputted by users to the system <br>
```outputs_full_match``` - To contain documents outputted by the system <br>
```outputs_partial_match``` - To contain documents outputted by the system <br>
```embeddings```- To contain embedding json files. This folder contains 3 sub folders.<br>
&nbsp;```|--sinhala```- To contains embedding of Sinhala documents<br>
&nbsp;```|--english```- To contain embeddings on English documents<br>
&nbsp;```|--parallel```- To contain embeddings of parallel documents<br>



### Embed Documents
#### Main option
You can find already embeded documents from <a href="">here[Add link]</a> <br>
You have to download and put those files to corresponding sub folder within the ```embeddings``` folder

#### Alternative option
If you want you can embed documents by yourself and use (but this is not recommended) <br>
Run below command to embed json list of documents using ```embedding_creator.py``` inside ```embedder``` folder.<br><br>
<free>
```python embedding_creator.py path/to/input_file.json file_type output_file_name```

Here ```file_type``` is ```si``` for sinhala documents, ```en``` for english documents and ```pa``` for parallel document parirs.<br>

The format of the input_file.json should be in following format.<br>
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

After this step, the folder structure of the ```Embeddings``` folder as follows.<br>
.<br>
|--embeddings<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--parallel<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- parallel_json_file_name1.json<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- parallel_json_file_name2.json<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- ...<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--sinhala<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- sinhala_json_file_name1.json<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- sinhala_json_file_name2.json<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- ...<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--english<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- english_json_file_name1.json<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- english_json_file_name2.json<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- ...<br>


### Preprocess Embedding Database

Run both of following commands to preprocess embedding database for indexing.<br><br>
```python db_split.py en```<br>
```python db_split.py si```

### Build Index Files
Run both of following commands to build index files.<br>(You should use same filename.py for both previous step and this step )<br><br>
```python indexing.py en```<br>
```python indexing.py si```

After this step, ```embeddings``` folder is no more needed.
### Run SimDocSin
Execute ```flask run``` within the webapp folder

### Contributors
Udhan Isuranga (udhanisuranga.16@cse.mrt.ac.lk) <br>
Janaka Sandaruwan (janakasadaruwan.16@cse.mrt.ac.lk) <br>
Udesh Athukorala (udeshathukorala.16@cse.mrt.ac.lk) <br>
