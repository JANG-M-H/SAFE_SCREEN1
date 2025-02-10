package com.example.yolo;

import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.widget.VideoView;
import androidx.appcompat.app.AppCompatActivity;

public class WorkActivity extends AppCompatActivity {

    private VideoView videoView;
    private Uri videoUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_work);

        videoView = findViewById(R.id.videoView);

        // 비디오 파일 경로 설정 (예: raw 리소스에 있는 video_sample.mp4 파일)
        String videoPath = "android.resource://" + getPackageName() + "/" + R.raw.tino4;
        videoUri = Uri.parse(videoPath);

        // VideoView에 재생할 비디오 설정
        videoView.setVideoURI(videoUri);

        // 비디오 재생 시작
        videoView.start();

        // 비디오 재생이 완료될 때 자동으로 다시 재생
        videoView.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mediaPlayer) {
                // 비디오 재생이 끝나면 다시 시작
                videoView.start();
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        // 액티비티 종료 시 VideoView 리소스 해제
        if (videoView != null) {
            videoView.stopPlayback();
        }
    }
}