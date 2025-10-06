from fastapi import FastAPI
import uvicorn
from store_app.api import mnist, fashion, cifar



store_app = FastAPI()
store_app.include_router(mnist.mnist_router)
store_app.include_router(fashion.fashion_router)
store_app.include_router(cifar.cifar_router)


if __name__ == '__main__':
    uvicorn.run(store_app,host='127.0.0.1',port=8005)