package com.example.carapp.View;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.EditText;

import com.example.carapp.Entites.User;
import com.example.carapp.R;
import com.example.carapp.View.baseProfile;

public class adminProfile extends baseProfile {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_admin_profile);
        this.context=getApplicationContext();

        this.user = (User) getIntent().getSerializableExtra("User");

        setBasicSettings();
        displayProfileData();
    }


    @Override
    protected void initViews() {

    }

    @Override
    protected void validateChange() {

    }

    @Override
    public void displayProfileData() {

    }

    @Override
    public void updateProfileData() {

    }
}
