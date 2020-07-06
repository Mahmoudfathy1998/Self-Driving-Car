package com.example.carapp.Controller.CallBacks;

import com.example.carapp.Entites.User;

public interface LoginCallBack {

    void onSuccess(User user);
    void onFailer(String error);

}
