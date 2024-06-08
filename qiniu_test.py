from qiniu import Auth, put_file
from qiniu import CdnManager

# 七牛云测试脚本
# 配置七牛云信息
access_key = ""
secret_key = ""
bucket_name = ""
bucket_url = ""
q = Auth(access_key, secret_key)
cdn_manager = CdnManager(q)

# 将本地图片上传到七牛云中
def upload_img(bucket_name, file_name, file_path):
    # generate token
    token = q.upload_token(bucket_name, file_name)
    put_file(token, file_name, file_path)

# 获得七牛云服务器上file_name的图片外链
def get_img_url(bucket_url, file_name):
    img_url = 'http://%s/%s' % (bucket_url, file_name)
    return img_url

# 需要上传到七牛云上面的图片的路径
image_up_name = "data/images/bus.jpg"
# 上传到七牛云后，保存成的图片名称
image_qiniu_name = "girl_2022.jpg"
# 将图片上传到七牛云,并保存成image_qiniu_name的名称
upload_img(bucket_name, image_qiniu_name, image_up_name)
# 取出和image_qiniu_name一样名称图片的url
url_receive = get_img_url(bucket_url, image_qiniu_name)
print(url_receive)

# 需要刷新的文件链接,由于不同时间段上传的图片有缓存，因此需要CDN清除缓存，
urls = [url_receive]
# URL刷新缓存链接,一天有500次的刷新缓存机会
refresh_url_result = cdn_manager.refresh_urls(urls)