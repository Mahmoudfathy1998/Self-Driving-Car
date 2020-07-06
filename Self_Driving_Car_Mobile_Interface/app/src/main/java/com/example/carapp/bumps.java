package com.example.carapp;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.Iterator;

public class bumps extends AppCompatActivity {
TextView msg;
    DatabaseReference myRef;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_bumps);
         myRef = FirebaseDatabase.getInstance().getReference();
        msg = findViewById(R.id.msg);
        RetreiveBumps();

    }
    void RetreiveBumps(){
        myRef.child("bumps").addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot dataSnapshot) {

                for (DataSnapshot locationSnapshot : dataSnapshot.getChildren()) {
                    String bump = locationSnapshot.getValue().toString();
                    //Log.d("bumpTag", "BumpState: " + bump); //log
                    msg.setText("CarState: " + bump);
                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError databaseError) {

            }
        });

    }
}
