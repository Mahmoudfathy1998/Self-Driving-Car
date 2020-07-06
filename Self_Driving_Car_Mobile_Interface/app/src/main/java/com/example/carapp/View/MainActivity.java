package com.example.carapp.View;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import com.example.carapp.Controller.CallBacks.LoginCallBack;
import com.example.carapp.Controller.UserController;
import com.example.carapp.Entites.User;
import com.example.carapp.R;

import java.io.Serializable;
import java.util.Random;

public class MainActivity extends AppCompatActivity {
    private Button loginBTN;
    private EditText emailET, passwordET;

    private UserController userController;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        userController = new UserController(this);

        loginBTN = findViewById(R.id.login);
        emailET = findViewById(R.id.loginemail);
        passwordET = findViewById(R.id.loginpassword);

        loginBTN.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
               login(emailET.getText().toString(), passwordET.getText().toString());
               // Toast.makeText(getApplicationContext(), getRandomString(10), Toast.LENGTH_LONG).show();
                //SendMail Passmail = new SendMail(v.getContext(), "Sara1603573@miuegypt.edu.eg", "TestMail", "This mail is from 7oda  ... hhh........");
                //Executing sendmail to send email
                //sm.execute();

            }
        });
    }

    public void login(String email, String password){

        userController.userLogin(email, password, new LoginCallBack() {
            @Override
            public void onSuccess(User user) {
                if(user.getUsertype().getId() == 1){ // admin
                    Intent intent1 = new Intent(getApplicationContext(), AdminActivity.class);
                    intent1.putExtra("User", (Serializable) user);
                    startActivity(intent1);
                }else{ // driverذذ1
                    Intent intent2 = new Intent(getApplicationContext(), DriverActivity.class);
                    intent2.putExtra("User", (Serializable) user);
                    startActivity(intent2);
                }
            }

            @Override
            public void onFailer(String error) {
                Toast.makeText(getApplicationContext(), error, Toast.LENGTH_LONG).show();
            }
        });

    }




}
