package com.example.carapp;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import com.example.carapp.View.MainActivity;

public class distance extends AppCompatActivity {
    private Button okBTN;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_distance);
        okBTN = (Button) findViewById(R.id.okBTN);
        okBTN.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                back();
            }
        });
    }
    public void back(){
        Intent intent = new Intent(this, MainActivity.class);
        startActivity(intent);
    }
}
