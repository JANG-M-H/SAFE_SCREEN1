package com.example.yolo;

import android.app.Activity;
import android.content.Intent;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity
{
    private static final int REQUEST_VIDEO_EDIT = 1; // 비디오 편집 요청을 식별하기 위한 상수 선언
    private static final int REQUEST_VIDEO_PICK = 2; // 비디오 선택 요청을 식별하기 위한 상수 선언
    private Uri videoUri; // 선택한 비디오의 URI를 저장

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button editVideoButton = findViewById(R.id.pickVideoButton); // '필터링 선택' 버튼
        Button galleryButton = findViewById(R.id.openGalleryButton); // '동영상 재생' 버튼

        editVideoButton.setOnClickListener( // '필터링 선택' 버튼 클릭 시에..
                new View.OnClickListener()
                {
                    @Override
                    public void onClick(View v)
                    {
                        openGalleryForVideoEdit(); // 'openGalleryForVideoEdit' 함수 호출
                    }
                }
        );

        galleryButton.setOnClickListener( // '동영상 재생' 버튼 클릭 시에..
                new View.OnClickListener()
                {
                    @Override
                    public void onClick(View v)
                    {
                        openGallery(); // 'openGallery' 함수 호출
                    }
                }
        );
    }

    private void openGalleryForVideoEdit() // 사용자가 비디오를 선택할 수 있도록, 갤러리 오픈
    {
        // intent 객체를 생성하고, 미디어 파일의 URI를 지정하여, 사용자가 항목을 선택할 수 있게함
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("video/*"); // 비디오 파일만 선택하도록 필터링
        startActivityForResult(intent, REQUEST_VIDEO_EDIT); // 명령을 구별하도록, 'REQUEST_VIDEO_EDIT' 요청
    }

    private void openGallery() // 사용자가 비디오를 선택할 수 있도록, 갤러리 오픈
    {
        // intent 객체를 생성하고, 미디어 파일의 URI를 지정하여, 사용자가 항목을 선택할 수 있게함
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("video/*"); // 비디오 파일만 선택하도록 필터링
        startActivityForResult(intent, REQUEST_VIDEO_PICK); // 명령을 구별하도록, 'REQUEST_VIDEO_PICK' 요청
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) // '어떤 요청인지', '영상이 잘 선택 됐는지', '어떤 영상인지'
    {
        super.onActivityResult(requestCode, resultCode, data); // 가져온 데이터로 기본 구현

        if (resultCode == RESULT_OK && data != null) // 사용자가 성공적으로 비디오를 선택한 경우
        {
            videoUri = data.getData(); // 선택한 비디오의 URI 가져옴

            if (videoUri != null) // 선택한 비디오가 null 값이 아니라면
            {
                if (requestCode == REQUEST_VIDEO_EDIT) // '필터링 선택' 요청인 경우
                {
                    uploadVideo(videoUri); // 'uploadVideo' 함수 호출
                    Intent intent = new Intent(MainActivity.this, WorkActivity.class); // UI에는 WorkActivity 띄어줌
                    startActivity(intent); // UI 전환
                }
                else if (requestCode == REQUEST_VIDEO_PICK) // '동영상 재생' 요청인 경우
                {
                    playVideo(videoUri); // 'playVideo' 함수 호출
                }
            }
        }
    }

    // 새로운 동영상 파일을 생성하는 메서드(서버에 업로드 하기 전에 동영상 설정 해줌)
    private File createVideoFile() throws IOException
    {
        String timeStamp = String.valueOf(System.currentTimeMillis()); // 현재 시스템 시간을 밀리초(milliseconds) 단위로 가져와 문자열로 변환
        String videoFileName = "VIDEO_" + timeStamp + "_"; // timeStamp를 기반으로 동영상 파일 이름을 생성
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_MOVIES); // 외부 저장소의 특정 디렉토리(Movies)를 가져옴
        File videoFile = File.createTempFile( // 임시 파일을 생성
                videoFileName,  /* prefix */
                ".mp4",         /* suffix */
                storageDir      /* directory */
        );

        return videoFile;
    }

    // 선택한 동영상을 서버에 업로드하고 처리된 결과를 받아오는 메서드
    private void uploadVideo(Uri videoUri) // URI Type의 videoUri를 받아옴
    {
        if (videoUri == null) // 받아온 videoUri가 없는 경우(동영상이 선택되지 않은 경우)
        {
            Toast.makeText(this, "Video not found", Toast.LENGTH_SHORT).show();
            return;
        }

        try
        {
            // [선택한 동영상 파일을 서버로 업로드](입력 스트림에서 데이터를 읽어 버퍼에 저장한 후, 출력 스트림으로 버퍼의 데이터를 써 줌)
            InputStream inputStream = getContentResolver().openInputStream(videoUri); // 선택한 비디오의 입력 스트림을 열어줌
            File videoFile = createVideoFile(); // 받아온 videoUri를 저장할 새로운 비디오 파일을 생성(별도의 메서드로 만들어줌)
            OutputStream outputStream = new FileOutputStream(videoFile); // 생성된 비디오 파일에 대한 출력 스트림을 열어줌

            byte[] buffer = new byte[8 * 1024]; // 입력 스트림에서 읽어온 데이터를 임시로 저장하기 위한 8K 바이트 배열 생성(나눠서 순차적으로 보내기 위함)
            int bytesRead; // 입력 스트림에서 실제로 읽어온 바이트 수를 저장하기 위한 변수 선언

            // 최대 8K 바이트만큼 데이터를 읽어와 버퍼에 저장하고, 더이상 읽을 데이터가 없다면 '-1'를 반환하여 루프 종료
            while ((bytesRead = inputStream.read(buffer)) != -1)
            {
                outputStream.write(buffer, 0, bytesRead); // 버퍼의 데이터를 출력 스트림으로 작성
            }
            outputStream.close(); // 출력 스트림을 닫음(파일에 대한 모든 쓰기 작업을 완료하고 리소스를 해제)
            inputStream.close(); // 입력 스트림을 닫음(파일에 대한 모든 읽기 작업을 완료하고 리소스를 해제)


            // [동영상 파일을 서버로 전송] 아래 세 줄은, 비디오 파일을 서버로 전송하는 데 필요한 모든 준비 작업을 수행
            RequestBody requestFile = RequestBody.create(MediaType.parse("video/*"), videoFile); // 비디오 파일을 HTTP 요청 본문으로 포함하기 위해 RequestBody 객체를 생성
            MultipartBody.Part videoPart = MultipartBody.Part.createFormData("video", videoFile.getName(), requestFile); // RequestBody에 데이터 이름과 파일 이름을 추가하여 HTTP 요청의 특정 형식으로 가공
            ApiService apiService = RetrofitClient.createService(ApiService.class); // Retrofit API 서비스 객체를 만듦(해당 객체로 서버 통신 진핸)

            // 서버에 비디오 가공 처리를 요청(비동기로 실행됨)
            apiService.processVideo(videoPart).enqueue(new Callback<ResponseBody>() // videoPart를 매개변수로 받아, 이를 서버에 전송
            {                                  // enqueue 메서드는 비동기 요청을 수행하도록 함. 요청이 완료되면 콜백이 호출 됨
                @Override
                public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) // 서버 응답이 성공적으로 수신되었을 때 호출 됨
                {             // call: 서버에 대한 요청을 나타내는 객체 / response: 서버로부터의 응답을 나타내는 객체
                    if (response.isSuccessful()) // 응답이 성공적(HTTP 상태 코드 200-299)인지 여부를 반환
                    {
                        try {
                            // 처리된 동영상 파일을 외부 저장소에 저장
                            // response.body().byteStream()를 통해 서버에서 받은 비디오 스트림을 가져옴과 동시에,
                            // saveVideoToExternalStorage 메서드를 호출하여 비디오 스트림을 외부 저장소에 저장
                            saveVideoToExternalStorage(response.body().byteStream(), "Yolotest.mp4");
                        } catch (IOException e) // 저장 중 예외가 발생한 경우
                        {
                            e.printStackTrace(); // 예외가 발생한 위치와 경로를 보여주는 스택 트레이스
                            Toast.makeText(MainActivity.this, "Failed to save video", Toast.LENGTH_SHORT).show();
                        }
                    }
                    else // 서버 응답이 실패한 경우..
                    {
                        Toast.makeText(MainActivity.this, "Failed to upload video", Toast.LENGTH_SHORT).show();
                    }
                }

                @Override
                public void onFailure(Call<ResponseBody> call, Throwable t) // 네트워크 오류나 다른 이유로 요청이 실패했을 때 호출
                {
                    Toast.makeText(MainActivity.this, "Network error: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                }
            });
        } catch (IOException e)
        {
            e.printStackTrace(); // // 예외가 발생한 위치와 경로를 보여주는 스택 트레이스
            Toast.makeText(this, "Failed to upload video", Toast.LENGTH_SHORT).show();
        }
    }

    // '동영상 재생' 버튼이 눌려서, 갤러리로 이동한 경우
    private void playVideo(Uri videoUri)
    {
        Intent intent = new Intent(Intent.ACTION_VIEW, videoUri); // 선택한 비디오를 재생
        intent.setDataAndType(videoUri, "video/*");
        startActivity(intent);
    }

    // 동영상 파일을 외부 저장소에 저장하는 메서드
    private void saveVideoToExternalStorage(InputStream inputStream, String fileName) throws IOException
    {
        // File directory = getExternalFilesDir(Environment.DIRECTORY_MOVIES);
        // 외부 저장소의 'Movies' 디렉토리 경로를 가져옵
        File directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES);

        if (!directory.exists()) // // 저장할 디렉토리가 없다면 생성합니다.
        {
            directory.mkdirs();
        }

        File videoFile = new File(directory, fileName); // 실제 동영상 파일 객체 videoFile을 생성
        OutputStream outputStream = new FileOutputStream(videoFile); // videoFile에 데이터를 쓸 outputStream을 생성

        // [아래 과정을 반복하면서 서버로부터 받은 영상을, Movies 폴더에 저장]
        byte[] buffer = new byte[8 * 1024]; // // 입력 스트림에서 읽어온 데이터를 임시로 저장하기 위한 8K 바이트 배열 생성(나눠서 순차적으로 보내기 위함)
        int bytesRead;

        while ((bytesRead = inputStream.read(buffer)) != -1) // inputStream에서 버퍼 크기만큼 데이터를 읽어옴
        {
            outputStream.write(buffer, 0, bytesRead); // 버퍼에 있는 데이터를 videoFile에 기록
        }
        outputStream.close();
        inputStream.close();

        // 동영상 파일을 미디어 스캐너에 등록하여 갤러리에 표시되도록 함
        // (Android 시스템이 해당 파일을 인식하고 갤러리 앱 등에서 표시할 수 있도록 해주는 과정이다)
        MediaScannerConnection.scanFile(this, new String[]{videoFile.getAbsolutePath()}, null, // 저장한 동영상 파일을 미디어 스캐너에 등록
                (path, uri) -> {
                    Log.i("MediaScanner", "Scanned " + path); // Log.i를 사용하여 스캔된 파일의 경로를 로그로 출력
                    Log.i("MediaScanner", "-> uri=" + uri); // Log.i를 사용하여 스캔된 파일의 uri를 로그로 출력
                });

        Toast.makeText(this, "비디오가 해당 위치에 저장되었습니다 ->" + videoFile.getAbsolutePath(), Toast.LENGTH_LONG).show();
        Intent intent = new Intent(MainActivity.this, MainActivity.class);
        startActivity(intent);
    }
}
