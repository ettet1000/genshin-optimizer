(this["webpackJsonpgenshin-optimizer"]=this["webpackJsonpgenshin-optimizer"]||[]).push([[4],{167:function(e,t,a){"use strict";a.d(t,"a",(function(){return j}));var n=a(162),i=a(106),r=a(169),o=a(76),c=a(166),l=a(16),s=a(60),u=a(61),v=a(183),d=a(165),f=a(195),p=a(207),m=a(170),h=a(29),g=a(26),L=a(192),b=a(189),y=a(202),O=a(90),_=a(9),S=a(186),j=function(){function e(){if(Object(s.a)(this,e),this instanceof e)throw Error("A static class cannot be instantiated.")}return Object(u.a)(e,null,[{key:"remove",value:function(e){var t=g.a.get(e);t&&(Object.values(t.equippedArtifacts).forEach((function(e){return h.a.moveToNewLocation(e,"")})),g.a.remove(e))}},{key:"getDisplayHeading",value:function(e,t,a){if("basicKeys"===e)return"Basic Stats";if("genericAvgHit"===e)return"Generic Optimization Values";if("transReactions"===e)return"Transformation Reaction";if(e.startsWith("talentKey_")){var n,i,r=e.split("talentKey_")[1];return null!==(n=null===t||void 0===t||null===(i=t.getTalent(r))||void 0===i?void 0:i.name)&&void 0!==n?n:r}if(e.startsWith("weapon_")){var o,c=e.split("weapon_")[1];return null!==(o=null===a||void 0===a?void 0:a.name)&&void 0!==o?o:c}return""}}]),e}();j.getElementalName=function(e){var t,a=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"";return(null===m.a||void 0===m.a||null===(t=m.a[e])||void 0===t?void 0:t.name)||a},j.getlevelKeys=function(){return Object.keys(p.b)},j.getlevelTemplateName=function(e){var t,a=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"";return(null===p.b||void 0===p.b||null===(t=p.b[e])||void 0===t?void 0:t.name)||a},j.getLevelString=function(e,t,a){var n=j.getStatValueWithOverride(e,t,a,"characterLevel");return j.getLevel(e.levelKey)===n?j.getlevelTemplateName(e.levelKey):"Lvl. ".concat(n)},j.getIndexFromlevelkey=function(e){return j.getlevelKeys().indexOf(e)},j.getLevel=function(e){var t,a=arguments.length>1&&void 0!==arguments[1]?arguments[1]:1;return(null===p.b||void 0===p.b||null===(t=p.b[e])||void 0===t?void 0:t.level)||a},j.getAscension=function(e){var t,a=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0;return(null===p.b||void 0===p.b||null===(t=p.b[e])||void 0===t?void 0:t.asend)||a},j.getTalentFieldValue=function(e,t){var a=arguments.length>2&&void 0!==arguments[2]?arguments[2]:{},n=arguments.length>3&&void 0!==arguments[3]?arguments[3]:"";return(null===e||void 0===e?void 0:e[t])?Object(_.h)(null===e||void 0===e?void 0:e[t],a):n},j.hasOverride=function(e,t){return"finalHP"===t?j.hasOverride(e,"hp")||j.hasOverride(e,"hp_")||j.hasOverride(e,"characterHP")||!1:"finalDEF"===t?j.hasOverride(e,"def")||j.hasOverride(e,"def_")||j.hasOverride(e,"characterDEF")||!1:"finalATK"===t?j.hasOverride(e,"atk")||j.hasOverride(e,"atk_")||j.hasOverride(e,"characterATK")||!1:!!(null===e||void 0===e?void 0:e.baseStatOverrides)&&t in e.baseStatOverrides},j.getBaseStatValue=function(e,t,a,n){var i=arguments.length>4&&void 0!==arguments[4]?arguments[4]:0,r=e.levelKey;return"specializedStatKey"===n?t.specializeStat.key:"specializedStatVal"===n?t.specializeStat.value[j.getIndexFromlevelkey(r)]:"weaponATK"===n?S.a.getWeaponMainStatValWithOverride(null===e||void 0===e?void 0:e.weapon,a):"characterLevel"===n||"enemyLevel"===n?j.getLevel(r):n.includes("enemyRes_")?10:n in p.c?p.c[n]:n in t.baseStat?t.baseStat[n][j.getIndexFromlevelkey(r)]:i},j.getStatValueWithOverride=function(e,t,a,n){var i,r,o=arguments.length>4&&void 0!==arguments[4]?arguments[4]:0;return j.hasOverride(e,n)?null!==(i=null===e||void 0===e||null===(r=e.baseStatOverrides)||void 0===r?void 0:r[n])&&void 0!==i?i:o:j.getBaseStatValue(e,t,a,n,o)},j.equipArtifacts=function(e,t){var a=g.a.get(e);if(a){var n=a.equippedArtifacts;O.h.forEach((function(a){var i,r,o=h.a.get(t[a]);if((null===o||void 0===o?void 0:o.location)!==e){var c=h.a.get(null===n||void 0===n?void 0:n[a]),l=null!==(i=null===o||void 0===o?void 0:o.location)&&void 0!==i?i:"";c&&h.a.moveToNewLocation(c.id,l),l&&g.a.equipArtifactOnSlot(l,a,null!==(r=null===c||void 0===c?void 0:c.id)&&void 0!==r?r:""),o&&h.a.moveToNewLocation(o.id,e)}})),g.a.equipArtifactBuild(e,t)}},j.calculateBuild=function(e,t,a,n){var i,r=arguments.length>4&&void 0!==arguments[4]?arguments[4]:0;if(e.artifacts)i=Object.fromEntries(e.artifacts.map((function(e,t){return[t,e]})));else{if(!e.equippedArtifacts)return{};i=Object.fromEntries(Object.entries(e.equippedArtifacts).map((function(e){var t=Object(l.a)(e,2),a=t[0],n=t[1];return[a,h.a.get(n)]})))}var o=j.createInitialStats(e,t,a);return o.mainStatAssumptionLevel=r,j.calculateBuildwithArtifact(o,i,n)},j.calculateBuildwithArtifact=function(e,t,a){var n,i=v.a.setToSlots(t),r=d.a.setEffectsStats(a,e,i),o=Object(_.f)(e);Object.values(t).forEach((function(e){e&&(o[e.mainStatKey]=(o[e.mainStatKey]||0)+v.a.mainStatValue(e.mainStatKey,e.numStars,Math.max(Math.min(o.mainStatAssumptionLevel,4*e.numStars),e.level)),e.substats.forEach((function(e){return e&&e.key&&(o[e.key]=(o[e.key]||0)+e.value)})))})),r.forEach((function(e){return o[e.key]=(o[e.key]||0)+e.value})),f.a.parseConditionalValues({artifact:null===o||void 0===o||null===(n=o.conditionalValues)||void 0===n?void 0:n.artifact},(function(e,t,a){var n,r,c=Object(l.a)(a,2)[1],s=e.setNumKey;if(!(parseInt(s)>(null!==(n=null===i||void 0===i||null===(r=i[c])||void 0===r?void 0:r.length)&&void 0!==n?n:0))){var u=f.a.resolve(e,o,t).stats;Object.entries(u).forEach((function(e){var t=Object(l.a)(e,2),a=t[0],n=t[1];return o[a]=(o[a]||0)+n}))}})),o.equippedArtifacts=Object.fromEntries(Object.entries(t).map((function(e){var t=Object(l.a)(e,2),a=t[0],n=t[1];return[a,null===n||void 0===n?void 0:n.id]}))),o.setToSlots=i;var c=Object(y.a)(null===o||void 0===o?void 0:o.modifiers);return Object(b.c)(c,o).formula(o),o},j.mergeStats=function(e,t){return t&&Object.entries(t).forEach((function(t){var a=Object(l.a)(t,2),n=a[0],i=a[1];if("modifiers"===n){var r;e.modifiers=null!==(r=e.modifiers)&&void 0!==r?r:{};var o,s=Object(c.a)(Object.entries(i));try{for(s.s();!(o=s.n()).done;){var u,v=Object(l.a)(o.value,2),d=v[0],f=v[1];e.modifiers[d]=null!==(u=e.modifiers[d])&&void 0!==u?u:{};var p,m=Object(c.a)(Object.entries(f));try{for(m.s();!(p=m.n()).done;){var h,g=Object(l.a)(p.value,2),L=g[0],b=g[1];e.modifiers[d][L]=(null!==(h=e.modifiers[d][L])&&void 0!==h?h:0)+b}}catch(y){m.e(y)}finally{m.f()}}}catch(y){s.e(y)}finally{s.f()}}else void 0===e[n]?e[n]=i:"number"===typeof e[n]&&(e[n]+=i)}))},j.createInitialStats=function(e,t,a){var c,s,u=e=Object(_.f)(e),v=u.characterKey,d=u.levelKey,m=u.hitMode,h=u.infusionAura,g=u.reactionMode,L=u.talentLevelKeys,b=u.constellation,y=u.equippedArtifacts,A=u.conditionalValues,w=void 0===A?{}:A,E=u.weapon,K=void 0===E?{key:""}:E,W=j.getAscension(d),k=["characterHP","characterATK","characterDEF","weaponATK","characterLevel","enemyLevel","physical_enemyRes_","physical_enemyImmunity"].concat(Object(o.a)(Object.keys(p.c))),V=Object.fromEntries(k.map((function(n){return[n,j.getStatValueWithOverride(e,t,a,n)]})));V.characterEle=t.elementKey,V.characterKey=v,V.hitMode=m,V.infusionAura=h,V.reactionMode=g,V.conditionalValues=w,V.weaponType=t.weaponTypeKey,V.tlvl=L,V.constellation=b,V.ascension=W,V.weapon=K,V.equippedArtifacts=y,["physical"].concat(Object(o.a)(O.d)).forEach((function(n){var i="".concat(n,"_enemyRes_");V[i]=j.getStatValueWithOverride(e,t,a,i),i="".concat(n,"_enemyImmunity"),V[i]=j.getStatValueWithOverride(e,t,a,i)}));var T=(null===(c=e)||void 0===c?void 0:c.baseStatOverrides)||{};Object.entries(T).forEach((function(e){var t=Object(l.a)(e,2),a=t[0],n=t[1];"specializedStatKey"!==a&&"specializedStatVal"!==a&&(V.hasOwnProperty(a)||(V[a]=n))}));var I=j.getStatValueWithOverride(e,t,a,"specializedStatVal"),M=j.getStatValueWithOverride(e,t,a,"specializedStatKey");for(var R in j.mergeStats(V,Object(r.a)({},M,I)),t.getTalentStatsAll(V).forEach((function(e){return j.mergeStats(V,e)})),V.tlvl){var x;V.tlvl[R]+=null!==(x=V["".concat(R,"Boost")])&&void 0!==x?x:0}var z=S.a.getWeaponSubstatKey(a);z&&j.mergeStats(V,Object(r.a)({},z,S.a.getWeaponSubstatValWithOverride(null===(s=e)||void 0===s?void 0:s.weapon,a))),j.mergeStats(V,a.stats(V));w.artifact;var H=w.weapon,q=Object(i.a)(w,["artifact","weapon"]);return f.a.parseConditionalValues(Object(n.a)(Object(n.a)({},K.key&&{weapon:Object(r.a)({},K.key,null===H||void 0===H?void 0:H[K.key])}),q),(function(e,t,a){if(f.a.canShow(e,V)){var n=f.a.resolve(e,V,t).stats;j.mergeStats(V,n)}})),V},j.getDisplayStatKeys=function(e,t){var a,i=e.characterKey,r=t.elementKey,c=["finalHP","finalATK","finalDEF","eleMas","critRate_","critDMG_","heal_","enerRech_","".concat(r,"_dmg_")],s=t.isAutoElemental;s||c.push("physical_dmg_");var u=Object(_.f)(b.a[r]),v=t.weaponTypeKey;if(u.includes("shattered_hit")||"claymore"!==v||u.push("shattered_hit"),null===(a=L.a.formulas.character)||void 0===a?void 0:a[i]){var d={};Object.entries(L.a.formulas.character[i]).forEach((function(t){var a=Object(l.a)(t,2),n=a[0],i=a[1];Object.values(i).forEach((function(t){if(t.field.canShow(e)){"normal"!==n&&"charged"!==n&&"plunging"!==n||(n="auto");var a="talentKey_".concat(n);d[a]||(d[a]=[]),d[a].push(t.keys)}}))}));var f=L.a.formulas.weapon[e.weapon.key];return f&&Object.values(f).forEach((function(t){if(t.field.canShow(e)){var a="weapon_".concat(e.weapon.key);d[a]||(d[a]=[]),d[a].push(t.keys)}})),Object(n.a)(Object(n.a)({basicKeys:c},d),{},{transReactions:u})}var p=[];if(s?"bow"===v&&p.push("".concat(r,"_charged_avgHit")):p.push("physical_normal_avgHit","physical_charged_avgHit"),p.push("".concat(r,"_skill_avgHit"),"".concat(r,"_burst_avgHit")),"pyro"===r){var m=[];m.push.apply(m,Object(o.a)(p.filter((function(e){return e.startsWith("".concat(r,"_"))})).map((function(e){return e.replace("".concat(r,"_"),"".concat(r,"_vaporize_"))})))),m.push.apply(m,Object(o.a)(p.filter((function(e){return e.startsWith("".concat(r,"_"))})).map((function(e){return e.replace("".concat(r,"_"),"".concat(r,"_melt_"))})))),p.push.apply(p,m)}else"cryo"===r?p.push.apply(p,Object(o.a)(p.filter((function(e){return e.startsWith("".concat(r,"_"))})).map((function(e){return e.replace("".concat(r,"_"),"".concat(r,"_melt_"))})))):"hydro"===r&&p.push.apply(p,Object(o.a)(p.filter((function(e){return e.startsWith("".concat(r,"_"))})).map((function(e){return e.replace("".concat(r,"_"),"".concat(r,"_vaporize_"))}))));return{basicKeys:c,genericAvgHit:p,transReactions:u}}},180:function(e,t,a){"use strict";var n={elements:{anemo:a.p+"static/media/Element_Anemo.f809fde3.png",cryo:a.p+"static/media/Element_Cryo.019d72f9.png",dendro:a.p+"static/media/Element_Dendro.8ee0f26d.png",electro:a.p+"static/media/Element_Electro.342332ac.png",geo:a.p+"static/media/Element_Geo.b7e865c6.png",hydro:a.p+"static/media/Element_Hydro.f2f8bd8a.png",pyro:a.p+"static/media/Element_Pyro.f65c2e38.png"},weaponTypes:{bow:a.p+"static/media/Weapon-class-bow-icon.b8e7b5ca.png",catalyst:a.p+"static/media/Weapon-class-catalyst-icon.2cbef800.png",claymore:a.p+"static/media/Weapon-class-claymore-icon.17418b20.png",polearm:a.p+"static/media/Weapon-class-polearm-icon.a4e7fffc.png",sword:a.p+"static/media/Weapon-class-sword-icon.4470b487.png"},resin:{fragile:a.p+"static/media/Item_Fragile_Resin.f9ec8223.png",condensed:a.p+"static/media/Item_Condensed_Resin.1cecf64a.png"},exp_books:{advice:a.p+"static/media/Item_Wanderer's_Advice.58c62cf7.png",wit:a.p+"static/media/Item_Hero's_Wit.a79e36d0.png",experience:a.p+"static/media/Item_Adventurer's_Experience.92b5d195.png"}};t.a=n},186:function(e,t,a){"use strict";a.d(t,"a",(function(){return r}));var n=a(60),i=a(201),r=function e(){if(Object(n.a)(this,e),this instanceof e)throw Error("A static class cannot be instantiated.")};r.getLevelName=function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"";return i.a[e]||t},r.getLevelIndex=function(e){return i.b.indexOf(e)},r.getWeaponMainStatVal=function(e,t){var a=arguments.length>2&&void 0!==arguments[2]?arguments[2]:0;return e.baseStats.main[r.getLevelIndex(t)]||a},r.getWeaponSubstatVal=function(e,t){var a,n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:0;return(null===(a=e.baseStats.sub)||void 0===a?void 0:a[r.getLevelIndex(t)])||n},r.getWeaponSubstatKey=function(e){var t,a=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"";return(null===(t=e.baseStats)||void 0===t?void 0:t.substatKey)||a},r.getWeaponTypeName=function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"";return i.c[e]||t},r.getWeaponTypeKeys=function(){return Object.keys(i.c)},r.getWeaponMainStatValWithOverride=function(e,t){var a=arguments.length>2&&void 0!==arguments[2]?arguments[2]:0;return(null===e||void 0===e?void 0:e.overrideMainVal)||r.getWeaponMainStatVal(t,null===e||void 0===e?void 0:e.levelKey,a)},r.getWeaponSubstatValWithOverride=function(e,t){var a=arguments.length>2&&void 0!==arguments[2]?arguments[2]:0;return(null===e||void 0===e?void 0:e.overrideSubVal)||r.getWeaponSubstatVal(t,null===e||void 0===e?void 0:e.levelKey,a)}},192:function(e,t,a){"use strict";a.d(t,"a",(function(){return o}));var n=a(60),i=a(9),r=Promise.all([a.e(5),a.e(20)]).then(a.bind(null,270)).then((function(e){o.formulas=e.default,Object(i.e)(e.default,[],(function(e){return"function"===typeof e}),(function(e,t){return e.keys=t}))})),o=function e(){if(Object(n.a)(this,e),this instanceof e)throw Error("A static class cannot be instantiated.")};o.formulas={},o.get=function(e){return r.then((function(){return Object(i.q)(o.formulas,e)}))}},201:function(e,t,a){"use strict";a.d(t,"b",(function(){return n})),a.d(t,"a",(function(){return i})),a.d(t,"c",(function(){return r}));var n=["L1","L5","L10","L15","L20","L20A","L25","L30","L35","L40","L40A","L45","L50","L50A","L55","L60","L60A","L65","L70","L70A","L75","L80","L80A","L85","L90"],i={L1:"Lvl. 1",L5:"Lvl. 5",L10:"Lvl. 10",L15:"Lvl. 15",L20:"Lvl. 20",L20A:"Lvl. 20/40",L25:"Lvl. 25",L30:"Lvl. 30",L35:"Lvl. 35",L40:"Lvl. 40",L40A:"Lvl. 40/50",L45:"Lvl. 45",L50:"Lvl. 50",L50A:"Lvl. 50/60",L55:"Lvl. 55",L60:"Lvl. 60",L60A:"Lvl. 60/70",L65:"Lvl. 65",L70:"Lvl. 70",L70A:"Lvl. 70/80",L75:"Lvl. 75",L80:"Lvl. 80",L80A:"Lvl. 80/90",L85:"Lvl. 85",L90:"Lvl. 90"},r={sword:"Sword",claymore:"Claymore",catalyst:"Catalyst",bow:"Bow",polearm:"Polearm"}},202:function(e,t,a){"use strict";a.d(t,"a",(function(){return c}));a(16);var n=a(76),i=a(189);function r(e){var t=new Set;return e(new Proxy({},{get:function(e,a,n){t.add(a)}}),new Proxy({},{get:function(e,a,n){t.add(a)}})),Object(n.a)(t)}var o=Object.freeze(Object.fromEntries(Object.keys(i.b).map((function(e){return[e,r(i.b[e])]}))));function c(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:Object.keys(i.d),a=new Set;return t.forEach((function(t){return l(t,e,a)})),Object(n.a)(a)}function l(e,t,a){var n,i;a.has(e)||(null===(n=o[e])||void 0===n||n.forEach((function(e){return l(e,t,a)})),Object.keys(null!==(i=t[e])&&void 0!==i?i:{}).forEach((function(e){return l(e,t,a)})),a.add(e))}},207:function(e,t,a){"use strict";a.d(t,"b",(function(){return n})),a.d(t,"c",(function(){return i})),a.d(t,"a",(function(){return r}));var n={L1:{name:"Lv. 1",level:1,asend:0},L20:{name:"Lv. 20",level:20,asend:0},L20A:{name:"Lv. 20/40",level:20,asend:1},L40:{name:"Lv. 40",level:40,asend:1},L40A:{name:"Lv. 40/50",level:40,asend:2},L50:{name:"Lv. 50",level:50,asend:2},L50A:{name:"Lv. 50/60",level:50,asend:3},L60:{name:"Lv. 60",level:60,asend:3},L60A:{name:"Lv. 60/70",level:60,asend:4},L70:{name:"Lv. 70",level:70,asend:4},L70A:{name:"Lv. 70/80",level:70,asend:5},L80:{name:"Lv. 80",level:80,asend:5},L80A:{name:"Lv. 80/90",level:80,asend:6},L90:{name:"Lv. 90",level:90,asend:6}},i={critRate_:5,critDMG_:50,enerRech_:100,stamina:100},r=["hp_","atk_","def_","eleMas","enerRech_","heal_","critRate_","critDMG_","physical_dmg_","anemo_dmg_","geo_dmg_","electro_dmg_","hydro_dmg_","pyro_dmg_","cryo_dmg_"]}}]);
//# sourceMappingURL=4.3cde8341.chunk.js.map