package com.example.carapp.Controller.CallBacks;

import com.example.carapp.Entites.User;

import java.util.ArrayList;

public interface SelectUsersListCallBack {

    void onSuccess(ArrayList<User> userArrayList);
    void onError(String reason);

}
