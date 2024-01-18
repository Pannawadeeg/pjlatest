package com.example.sosad;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.widget.Button;
public class MainActivity extends AppCompatActivity {

        public static int TIME_OUT = 3000;
        @Override
        protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_main);
            //ตั้งเวลาเปลี่นหน้าจอ
            new Handler().postDelayed((Runnable) new Runnable() {
                public void run() {
                    Intent homeIntent = new Intent(MainActivity.this, MainActivity2.class);
                    startActivity(homeIntent);
                    finish();
                }
            }, TIME_OUT);
        }

    }