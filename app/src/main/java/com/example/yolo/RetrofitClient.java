package com.example.yolo;

import okhttp3.OkHttpClient;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import java.util.concurrent.TimeUnit;

public class RetrofitClient
{
    private static final String BASE_URL = "http://172.30.1.47:8080//";
    private static Retrofit retrofit;

    public static Retrofit getInstance()
    {
        if (retrofit == null)
        {
            OkHttpClient client = new OkHttpClient.Builder()
                    .connectTimeout(60, TimeUnit.SECONDS) // 연결 시간 초과 설정 (예: 60초)
                    .readTimeout(900, TimeUnit.SECONDS) // 읽기 시간 초과 설정 (예: 600초)
                    .addInterceptor(chain -> {
                        okhttp3.Request request = chain.request().newBuilder()
                                .header("Connection", "close")
                                .build();
                        return chain.proceed(request);
                    })
                    .build();

            retrofit = new Retrofit.Builder()
                    .baseUrl(BASE_URL)
                    .client(client) // 위에서 생성한 OkHttpClient를 사용
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
        }
        return retrofit;
    }

    // Coroutine을 사용하여 비동기적으로 서버에 요청을 보내는 함수
    public static <T> T createService(Class<T> serviceClass)
    {
        return getInstance().create(serviceClass);
    }
}
