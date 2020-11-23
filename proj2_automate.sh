
#!/bin/sh





path = "/users/pgrad/kumarv2/Captcha"


if not [-d "$path"]; then
        git clone https://github.com/vivek5559/Captcha.git
else
        git pull
fi

cd Captcha

python3 class.py --model-name model_char --len-model-name model_len  --captcha-dir Project2_test/ --output vivek.txt --symbols Symbols_new.txt

git add vivek.txt

git commit -m upload

git push origin master


































