from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
import os
import sys
sys.path.insert(1, '../../SimDocSin/')
from webapp_controller.main_app_controller import main ##This function takes EN doc as source and output to similar SI doc
from webapp_controller.partial_match_app_controller import main_partial
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = '12345678'

UPLOAD_FOLDER = '../../SimDocSin/inputs'
OUTPUT_FOLDER = '../../SimDocSin/outputs_full_match'
OUTPUT_FOLDER_PARTIALS = '../../SimDocSin/outputs_partial_match'
ALLOWED_EXTENSIONS = {'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['OUTPUT_FOLDER_PARTIALS'] = OUTPUT_FOLDER_PARTIALS

'''
@app.route('/')
def index():

    return render_template('index.html')#, posts=posts)
'''

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=('GET', 'POST'))
def index():

    return render_template('index.html')

@app.route('/partial_match', methods=('GET', 'POST'))
def partial_match():
    return render_template('partial_match.html')

@app.route('/search_partial_match', methods=('GET', 'POST'))
def search_partial_match():

    lang = request.form['lang']

    similarity_level = request.form['lvl']

    minimum_length = request.form['min_len']

    #print(lang,flush=True)
    
    if lang == None:
        lang = 'en'
    
    if request.method == 'POST':

        
        if request.form['action'] == 'Submit Text':
            #results = []
            sources = []
            content = request.form['content']

            if content=='' or content==None:
                return jsonify({"results":'error',"error":'Content is required'})
            else:
                sources.append(content)
                results, resultsNotAvailable = main_partial(sources, similarity_level, minimum_length, lang) ##This function takes EN doc as source and output to similar SI doc
                
                for result in results:
                    for i, target_id in enumerate(result['documents_targets_ids']):
                        output_file = open(os.path.join(app.config['OUTPUT_FOLDER_PARTIALS'], str(target_id)+'.txt'),'w+', encoding='utf-8')
                        output_file.write(result['documents_targets_belong_to'][i])
                        output_file.close()
                
                return jsonify({"results":results, 'resultsNotAvailable':resultsNotAvailable})
                
        elif request.form['action'] == 'Submit File':
            #results=[]
            sources = []
             
            file = request.files.get('myFile')#request.files['file']
            
            if file == None:
                return jsonify({"results":'error',"error":'No file selected'})

            if file and allowed_file(file.filename):
                
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
                #content = ''
                
                try:
                    source_file = open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'r',encoding='utf-8')
                    content = source_file.read()
                    sources.append(content)
        
                except IOError:
                    return jsonify({"results":'error',"error":'File reading error'})

                except UnicodeDecodeError:
                    return jsonify(
                        {"results": 'error', "error": 'File contains undecodable characters. Try to submit as a text.'})
                
                #result = {}
                #result['source'] = content
                #result['target'] = main(content, lang) ##This function takes EN doc as source and output to similar SI doc

                #results.append(result)
                results, resultsNotAvailable = main_partial(sources, similarity_level, minimum_length, lang) ##This function takes EN doc as source and output to similar SI doc
                
                for result in results:
                    for i, target_id in enumerate(result['documents_targets_ids']):
                        output_file = open(os.path.join(app.config['OUTPUT_FOLDER_PARTIALS'], str(target_id)+'.txt'),'w+', encoding='utf-8')
                        output_file.write(result['documents_targets_belong_to'][i])
                        output_file.close()

                
                return jsonify({"results":results, 'resultsNotAvailable':resultsNotAvailable})
           
    return render_template('partial_match.html')


@app.route('/about', methods=('GET', 'POST'))
def get_info():
           
    return render_template('about.html')

@app.route('/full_match', methods=('GET', 'POST'))
def full_match():

    similarity_level = request.form['lvl']
    
    lang = request.form['lang']#request.form.get('lang')
    
    if lang == None:
        lang = 'en'
    
    if request.method == 'POST':

        
        if request.form['action'] == 'Submit Text':
            results = []
            sources = []
            content = request.form['content']

            if not content:
                #flash('Content is required!')
                return jsonify({"results":'error',"error":'Content is required'})
            else:
                sources.append(content)
                #print(sources)
                #print(similarity_level)
                results = main(sources, similarity_level, lang) ##This function takes EN doc as source and output to similar SI doc
                
                for result in results:
                  for target in result['target']:
                    if target[0]>=0 :  
                        output_file = open(os.path.join(app.config['OUTPUT_FOLDER'], str(target[0])+'.txt'),'w+', encoding='utf-8')
                        output_file.write(target[1])
                        output_file.close()

                return jsonify({"results":results})
        
        elif request.form['action'] == 'Submit Files':

            #results = []
            
            #uploaded_files = request.files.getlist("file[]")
            file=request.files.get('myFile')

            sources = []
            
            #for file in uploaded_files:
            #print(file)
                
            if file == None:
                #flash('No selected file')
                return jsonify({"results":'error',"error":'No file selected'})#redirect(request.url)
            if file and allowed_file(file.filename):
                
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
                #content = ''
                
                try:
                    source_file = open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'r',encoding='utf-8')#, errors='ignore')
                    content = source_file.read()

                    sources.append(content)
        
                except IOError:
                    #flash('File reading error')
                    return jsonify({"results":'error',"error":'File reading error'})

                except UnicodeDecodeError:
                    return jsonify(
                        {"results": 'error', "error": 'File contains undecodable characters. Try to submit as a text.'})

                    #result = {}
            #result['source'] = content
            #result['target'] = main(content, lang) ##This function takes EN doc as source and output to similar SI doc

            #results.append(result)
            #print(sources)
            
            results = main(sources, similarity_level, lang) ##This function takes EN doc as source and output to similar SI doc
            #i = 0
            for result in results:
              for target in result['target']:
                if target[0]>=0 :  
                    output_file = open(os.path.join(app.config['OUTPUT_FOLDER'], str(target[0])+'.txt'),'w+', encoding='utf-8')
                    output_file.write(target[1])

                    output_file.close()
                #i+=1
    
            return jsonify({"results":results})#render_template('index.html', results=results,)
           
    return render_template('index.html')
           

@app.route('/get_matching_docs',methods = ['POST'])
def get_matching_docs():

    lang = 'en'
    similarity_level = 0
    file = None

    if 'lang' in request.form :
        lang = request.form['lang']
        if(lang!='si' and lang!='en'):
            return jsonify({"error":"Value of 'lang' parameter invalid"})
    
    if 'lvl' in request.form :
        lvl = request.form['lvl']
        if(lvl.isdigit()):
            similarity_level = max( 1, min(5, int(lvl))) - 1
        else:
            return jsonify({"error":"Value of 'lvl' parameter invalid"})

    if 'file' in request.files:
        file = request.files['file']
    else:
        return jsonify({"error":"No file selected"})
    
    sources = []
            
    if file == None:
        return jsonify({"error":'No file selected'})
    if file and allowed_file(file.filename):    
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        try:
            source_file = open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'r')
            content = source_file.read()
            sources.append(content)
        except IOError:
            return jsonify({"error":'File reading error'})

        except UnicodeDecodeError:
            return jsonify({"error": 'File reading error. File cotains undecodable character/s'})
    
    results = main(sources, similarity_level, lang) 
      
    return jsonify({"result":results})

@app.route('/get_partial_matchings',methods = ['POST'])
def get_partial_matchings():

    lang = 'en'
    similarity_level = 1
    minimum_length = '1'
    file = None

    if 'lang' in request.form :
        lang = request.form['lang']
        if(lang!='si' and lang!='en'):
            return jsonify({"error":"Value of 'lang' parameter invalid"})
    
    if 'lvl' in request.form :
        lvl = request.form['lvl']
        if(lvl.isdigit()):
            similarity_level = max( 1, min(5, int(lvl)))
        else:
            return jsonify({"error":"Value of 'lvl' parameter invalid"})

    if 'min_len' in request.form:
        min_len = request.form['min_len']
        if (min_len.isdigit()):
            minimum_length = min_len
        else:
            return jsonify({"error": "Value of 'min_len' parameter invalid"})

    if 'file' in request.files:
        file = request.files['file']
    else:
        return jsonify({"error":"No file selected"})
    
    sources = []
            
    if file == None:
        return jsonify({"error":'No file selected'})

    if file and allowed_file(file.filename):
        
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
     
        try:
            source_file = open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'r')
            content = source_file.read()
            sources.append(content)

        except IOError:
            return jsonify({"error":'File reading error'})

        except UnicodeDecodeError:
            return jsonify({"error": 'File reading error. File contains undecodable character/s'})
    
        results, resultsNotAvailable = main_partial(sources, similarity_level, minimum_length, lang) 
        if(resultsNotAvailable):
            return jsonify({"result":'No partially matching documents'})
        else:
            output=[]
            for result in results:
                temp = {}
                partial_source = result['matching_partal_source']
                partial_target = result['matching_partal_target']
                source_txt = ' '.join(i for i in partial_source)
                target_txt = '. '.join(i for i in partial_target)
                temp['matching_partial_source'] = source_txt
                temp['matching_partial_target'] = target_txt
                output.append(temp)

            return jsonify({"result":output})