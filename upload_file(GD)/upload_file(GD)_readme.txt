폴더 구조(플라스크 웹 서버 포함)
devil_wears_VITON
	viton_webapp
		web_main.py: flask 서버 구현
		static: 웹 관련 폴더(데이터, CSS 등)
		templetes: 웹 관련 폴더(html)
		viton: 이하 폴더 구조는 다른 사람들과 동일
			ACGPN_org
			detectron2
			runs
			sample
			...
			
파일 경로 및 올라오는 파일
웹 관련
devil_wears_VITON/viton_webapp/web_main.py
devil_wears_VITON/viton_webapp/templetes/mainpage.html
devil_wears_VITON/viton_webapp/templetes/inf_page.html
devil_wears_VITON/viton_webapp/viton/ACGPN_org/ACGPN_inference/test_init.py
devil_wears_VITON/viton_webapp/viton/detectron2/make_inference_dataset_api_run.py

devil_wears_VITON/viton_webapp/viton/detectron2/make_inference_dataset_decompo.py의 경우 수정할 부분이 없었기 때문에 따로 올리지 않았습니다.(해당 파일은 ipynb로 실행 및 설치)