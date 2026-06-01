// src/services/aiService.js

/**
 * Mock AI Service for the Sinhala Mental Health Chatbot.
 * In production, this will connect to the custom SLM via API.
 */

const delay = (ms) => new Promise(res => setTimeout(res, ms));

export const getSuggestion = async (question, answer, context) => {
    await delay(1000); // Simulate network latency

    const suggestions = [
        "ඔබගේ පිළිතුරට අනුව, ඔබ හොඳින් විවේක ගැනීම වැදගත් බව පෙනේ.",
        "මෙය සාමාන්‍ය තත්වයකි. දිනපතා භාවනා කිරීමෙන් මනස සන්සුන් කරගත හැක.",
        "ඔබගේ කලබලකාරී බව අඩු කර ගැනීමට ගැඹුරු හුස්ම ගැනීමේ ව්‍යායාම (Deep Breathing) කරන්න.",
        "වඩාත් යහපත් මානසික සෞඛ්‍යයක් සඳහා පවුලේ අය හෝ මිතුරන් සමඟ කතාබහ කිරීම සුදුසුය.",
        "ඔබගේ සිතුවිලි ලිහිල් කිරීම සඳහා ඔබේ ප්‍රියතම විනෝදාංශයක නිරත වීම වඩාත් යහපත් වේ.",
        "කෙටි ඇවිදීමක් හෝ ව්‍යායාම කිරීම මඟින් ඔබේ මනස සැහැල්ලු කරගත හැක.",
        "නින්දට පෙර තිර (Screens) භාවිතය අඩු කිරීමෙන් යහපත් නින්දක් ලබා ගත හැක."
    ];
    
    const randomIndex = Math.floor(Math.random() * suggestions.length);
    return suggestions[randomIndex];
};

export const getChatResponse = async (chatMessage, questionnaireContext, chatHistory) => {
    await delay(1500);
    
    // Check for some keywords
    if (chatMessage.includes("බය") || chatMessage.includes("භය")) {
        return "ඔබ තුළ යම් බියක් ඇති බව මට තේරෙනවා. ප්‍රශ්නාවලියේදී ඔබ කලබලකාරී බවක් පෙන්නුම් කළා. ගැඹුරු හුස්ම ගැනීමෙන් මෙය පාලනය කළ හැක. මේ බිය ඇතිවීමට විශේෂ හේතුවක් තිබේද?";
    }

    if (chatMessage.includes("නින්ද")) {
        return "නින්ද නොයාම මානසික පීඩනයේ ලක්ෂණයක් විය හැක. පෙර යෝජනා කළ පරිදි නින්දට පෙර කෝපි බීමෙන් වැළකී සිටීම උත්සාහ කරන්න.";
    }

    return "ඔබ පැවසූ කරුණ මට වැටහෙනවා. ඔබගේ පෙර පිළිතුරු වලට අනුව ඔබ යම් පීඩනයකින් සිටින බව පෙනෙන බැවින්, සන්සුන්ව සිටීමට උත්සාහ කරන්න. ඔබට වෙනත් ගැටළු ඇත්ද?";
};

export const getFinalSummary = async (questionnaireContext, chatHistory) => {
    await delay(1500);
    return "සම්පූර්ණ සංවාදයට අනුව මාගේ ප්‍රධාන යෝජනා:\n\n1. දිනපතා පැය 7-8 ක නින්දක් ලබා ගැනීමට උත්සාහ කරන්න.\n2. අධික පීඩනය හෝ කලබලකාරී බවක් දැනෙන සෑම අවස්ථාවකදීම ගැඹුරු හුස්ම ගැනීමේ ව්‍යායාම කරන්න.\n3. සිතට වද දෙන සිතුවිලි ඇත්නම් එය විශ්වාසවන්ත අයෙකු සමග බෙදා ගන්න.\n4. ඔබට තවදුරටත් මෙම අපහසුතා පවතී නම් වෘත්තීය මනෝවිද්‍යා උපදේශනයක් (Counseling) ලබා ගැනීමට පසුබට නොවන්න.";
};
