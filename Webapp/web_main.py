# Flask import, Flask의 인스턴스가 WSGI서버가 된다.
# 웹에서는 \/를 특별히 잘 구분하자
from flask import Flask, url_for, render_template, request
from werkzeug.utils import secure_filename
import os
import sys



# __name__의 이름으로 Flask instance를 생성한다. 이처럼 보통 모듈의 이름을 넣는다고 한다.
app= Flask(__name__)

## 아래는 코드 중 모델 초기화 코드가 삽입될 예정이다.
## 코드를 초기화하는데도 시간이 걸리기 때문에,
## 초기화를 먼저 시켜놓고 사용자의 요청이 올 때마다 inference를
## 진행하여 응답시간을 최소화시킬 예정이다.
##imp_path1=os.path.join(os.path.abspath('.'), 'viton\ACGPN_org\ACGPN_inference')
##sys.path.append(imp_path1)
##imp_path2=os.path.join(os.path.abspath('.'), 'viton\detectron2')
##sys.path.append(imp_path1)

## 광석님 보내주신 코드 임포트 및 초기화
##from viton.ACGPN_org.ACGPN_inference import infer_man
##model, opt=infer_man.initialize_Model()

## 민석님 보내주신 코드 임포트 및 초기화
## from viton.detectron2 import make_inference_dataset_api_run as data_man
## init_del(pth='viton/detectron2/datasets/new_datasets') #파일 삭제
## predictor=init_setup(pth='viton/detectron2')


# 아래처럼 두 url모두 하나의 함수에서 실행되도록 할 수 있다.
# render template는 html화면을 띄워주며, html은 template폴더에 저장된 것을 사용한다.(아마 경로 지정 속성도 있겠지??)
@app.route('/')
def main_page_on(name=None):
    return render_template('mainpage.html', name=name)


# inf_page에서 불러오는 함수, 상의, 하의 중 어떤 값을 합성할지 정한다.
@app.route('/result', methods=['POST'])
def inf_result():
    # type='hidden'이든, type='radio'든 form에 접근하는 방법은 매한가지다
    txt=request.form["select_cloth"]
    print('이러한 문장이...', txt)
    cloth1=request.form["cloth_part1"]
    print(cloth1)
    cloth2=request.form["cloth_part2"]
    print(cloth2)

    ## 민석님이 주실 inference함수를 실행시켜야 하며
    ## 실행시킨 함수의 결과가 세번째('IMG/person/b_girl.png'값이 들어간 위치에)에
    ## 들어가게 될 것이다.
    ## 민석님 디텍트론 인퍼런스 등 들어갈 자리
    ## api_one(origin_dir, img_dir, img_name, predictor, pth='viton/detectron2')
    ## api_two(origin_dir, img_clothes_dir, img_name, pth='viton/detectron2')

    return render_template('inf_page.html', data_list=['IMG/person/IU1.jpg', 'IMG/cloth/IU2.jpg', 'IMG/person/b_girl.png'])
    # return render_template('inf_page.html', imglist)



@app.route('/apply', methods=['GET', 'POST'])
def imgsave_for_inf(name=None):

    upload_dir_person=os.path.join(os.getcwd(),'static\IMG\person')
    upload_dir_cloth=os.path.join(os.getcwd(),'static\IMG\cloth')
    
    if request.method== 'POST':

        # 사람 이미지 저장
        print(request.files['person'])
        img_file1=request.files['person']
        upload_file1=os.path.join(upload_dir_person, img_file1.filename)
        # img_file1.save(upload_dir,secure_filename(img_file1.filename))
        print(upload_file1)
        img_file1.save(upload_file1)

        # 의상 이미지 저장, 사람 이미지와 동일함()
        print(request.files['cloth'])
        img_file2=request.files['cloth']
        upload_file2=os.path.join(upload_dir_cloth, img_file2.filename)
        print(upload_file2)
        img_file2.save(upload_file2)

        # 저장한 이미지 path혹은 이미지 인스턴스 보내서 돌리기(민석님 코드 오면 실행)
        # if 이미지가 상하의 다 있다면: 상하의 다 inf_page에 render_templete
        # elif 이미지가 상의나 하의만 있다면: inf_page에 있는 상의나 하의만 보내기
        ## 광석님 inference 들어갈 자리
        ##infer_man.generate_result(model, opt)

        
        # return render_template('inf_page.html', data_list=[upload_file1, upload_file2, upload_file1])
        return render_template('inf_page.html', data_list=['IMG/person/b_girl.png', 'IMG/cloth/b_girl2.png', 'IMG/person/IU1.jpg'])
        # return "파일업로드 완료!"
    else:
        return render_template('mainpage.html', name=name)
    
    return 'file uploaded successfully'





# 로컬 서버를 실행시킴
# 만약 주 실행 파일이 이 파일이라면,(== os에서 처음으로 실행시킨 파일이 이 파일이라면)
# __name__은 '__main__'이 되고
# if 문은 참이기 때문에 app.run()이 실행된다.
# app.run()은 모종의 단계를 거쳐 서버를 실행시킨다.
# 참고로, 이 상태에서는 local에서만 접근이 가능하다.
# 디버그 모드(기본 설정이라고 한다.)를 해제하거나 다른 네트워크에서 접근이 가능하도록 하려면
# app.run(host='0.0.0.0')을 실행시키면 된다.
# debug 모드는 파일을 저장하거나 주변 환경이 변하면(어떤 변화인지는 아직 잘 모르겠다)
# 실시간으로 코드를 재실행하여(당연히 서버도 재시작하게 됨) 
# 수정된 부분을 반영하는 역할을 한다. 
# app.run(debug=True)로도 적용 가능하다.
if __name__ == '__main__':
    app.debug= False
    app.run(host='0.0.0.0')






################################# 아래는 불필요한 내용(기본적인 학습을 위해 사용한 코드, 서버에 올릴 때 지울 것)###################




# 데코레이터, 어떤 URL이 우리가 작성한 함수를 실행시키는지 알려준다.
# 데코레이터를 이용한 주소 지정이라 보면 될듯
# 만약 인자로 "/main"을 주고, http://127.0.0.1:5000 를 주소로 한다면
# http://127.0.0.1:5000/main 에서 hello_world()가 실행된다.
# @app.route("/")
# def hello_world():
#     return 'Hello World!'



# app.route에 아래처럼 <username>를 넣으면
# url에 url/user/username으로 접근할 수 있으며
# 이렇게 들어간 username은 인자로 사용이 가능한 것으로 보인다
# 만약 함수 선언부의 매개변수에 username이 들어가지 않으면 
# url/user/username으로 접근시 type error(404에러와는 다르게 통신 자체는 200이지만 그 내부에서 error가 뜬다.)
# @app.route("/user/<username>")
# def show_user_profile(username):
#     return 'User %s' % username

# <int: post_id>처럼 주소 뒤에 오는 값에 대해 converter를 사용할 수 있다.
# (보통 이런 경우 post_id에 들어가는 type은 무조건 str이라고 생각했는데
# 그런 부분을 잘 고려한 것으로 보인다.)
# converter는 int, float, path가 존재하며, path의 경우 기본값과 비슷하지만 슬레시가 들어간다고 한다.
# @app.route("/post/<int:post_id>")
# def show_post(post_id):
#     return 'Post %d' % post_id

# app.test_request_context()는 뭐하는지 잘 모르겠다만...
# WSGI 환경을 조성한다고 하며, 이는 flask의 특이한 환경
# (아래처럼 변수를 통하지 않고 함수 이름으로 접근한다든지)
# 을 만들어주는 역할을 하는 것으로 보인다.
# url_for()는 url을 출력하는 함수다.
# 전달 인자는 함수의 이름이고, 필요한 경우 필요한 인자도 조합할 수 있다.
# 또한 원하는 문자열도 조합하여 url화 시킬 수 있는 것 같다.
# with app.test_request_context():
#     print(url_for('hello_world'))
#     print(url_for('show_user_profile', username='John Doe'))
#     print(url_for('show_user_profile', username='John Doe', next='/'))
#     print(url_for('show_post', post_id=1024))



# def do_the_login(): pass
# def show_the_login_form(): pass

# GET과 POST에 따라 나누어 처리 가능
# GET은 주로 정보를 얻기 위해 사용자가 서버에 보낼 목적으로 생성하는 요청(http method)이며,
# POST는 주로 어떤 정보를 사용자가 서버에 전달하기 위한 목적으로 생성하는 요청(http method)다.
# 아래 코드는 valid_login과 log_the_user_in이라는 가상의 함수를 작동시킨다 가정할 때
# request 객체에서 값들을 참조하는 것을 보여준다.
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error= None

#     if request.method=='POST': # request는 decorator에서 처리하는 걸까?
#         if valid_login(request.form['username'],
#                        request.form['password']):
#             return log_the_user_in(request.form['username'])
        
#         else:
#             error= "Invalid username/password"
    
#     return render_template('login.html', error=error)


# with app.test_request_context('/hello', method='POST'):
#     # now you can do something with the request until the
#     # end of the with block, such as basic assertions:
#     assert request.path == '/hello'
#     assert request.method == 'POST'